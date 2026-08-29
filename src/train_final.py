"""
Train the final multi-modal model with a 2-stage fine-tuning schedule.

  Stage 1 (epochs 1..freeze_epochs): backbone frozen, only the head trains at
    --lr_head. Lets the fresh head settle before it can perturb the backbone.
  Stage 2 (rest): backbone unfrozen, differential LRs (--lr_backbone vs
    --lr_head) with a linear warmup into cosine decay.

Reads metadata_clean.csv from step2_make_folds.py. Selects the best epoch on
validation ROC-AUC.

    python src/train_final.py --backbone tf_efficientnet_b4_ns --epochs 25
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from src.dataset import MelanomaDataset, check_images_exist
    from src.models.final_model import SkinMelanomaFinalModel
    from src.losses.focal_loss import BinaryFocalLoss
    from src.metrics import evaluate_predictions, find_best_threshold
except ImportError:
    # Falls back here when run as `python src/train_final.py`, which puts
    # this file's own directory (src/) on sys.path instead of the repo root.
    from dataset import MelanomaDataset, check_images_exist
    from models.final_model import SkinMelanomaFinalModel
    from losses.focal_loss import BinaryFocalLoss
    from metrics import evaluate_predictions, find_best_threshold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Must match SEX_CATEGORIES / SITE_CATEGORIES in step2_make_folds.py.
NUM_SEX_CATEGORIES = 3   # female, male, unknown
NUM_SITE_CATEGORIES = 7  # 6 body sites + unknown

# Raw metadata columns the model's embedding expects, in order:
# metadata[:,0]=sex, [:,1]=site, [:,2]=age.
META_COLS = ['sex_enc', 'site_enc', 'age_norm']


def set_seed(seed):
    """Same seed everywhere, so two runs of the same config are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_metadata_columns(df):
    """The embedding layer wants the raw label codes (sex_enc/site_enc) plus
    age_norm, not one-hot vectors."""
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata csv is missing columns {missing}; expected {META_COLS}")
    return df, list(META_COLS)


def build_augmentation():
    """Training-only augmentation. Normalisation and resize are handled in
    MelanomaDataset, not here."""
    import albumentations as A

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-30, 30), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ])


def build_loss(args, train_df, device):
    if args.loss == 'focal':
        print(f"  loss: BinaryFocalLoss(alpha={args.focal_alpha}, gamma={args.focal_gamma})")
        return BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    n_pos = int(train_df['target'].sum())
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"  loss: BCEWithLogitsLoss(pos_weight={pos_weight.item():.2f})")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# Mixup / CutMix. Images and targets are mixed with a shuffled copy of the
# batch; metadata is left index-aligned (categoricals don't interpolate).

def mixup_data(images, targets, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, targets, targets[perm], lam


def cutmix_data(images, targets, alpha):
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(images.size(0), device=images.device)
    _, _, H, W = images.shape
    cut_rat = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = int(torch.randint(H, (1,))), int(torch.randint(W, (1,)))
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))  # actual pasted area, not the sampled lam
    return mixed, targets, targets[perm], lam


def maybe_mixup_cutmix(images, targets, mixup_alpha, cutmix_alpha, prob):
    if prob <= 0 or torch.rand(1).item() > prob:
        return images, targets, targets, 1.0
    have_mixup, have_cutmix = mixup_alpha > 0, cutmix_alpha > 0
    if not have_mixup and not have_cutmix:
        return images, targets, targets, 1.0
    use_cutmix = have_cutmix and (not have_mixup or torch.rand(1).item() < 0.5)
    if use_cutmix:
        return cutmix_data(images, targets, cutmix_alpha)
    return mixup_data(images, targets, mixup_alpha)


def set_train_mode(model, backbone_frozen):
    """Freezing the backbone means .eval() too, otherwise its BatchNorm running
    stats keep updating even with requires_grad off."""
    model.train()
    if backbone_frozen:
        model.backbone.eval()


def build_stage1_optimizer(model, lr_head, weight_decay):
    for p in model.backbone_parameters():
        p.requires_grad_(False)
    return torch.optim.AdamW(model.head_parameters(), lr=lr_head, weight_decay=weight_decay)


def build_stage2_optimizer(model, lr_backbone, lr_head, weight_decay):
    for p in model.backbone_parameters():
        p.requires_grad_(True)
    return torch.optim.AdamW([
        {'params': model.backbone_parameters(), 'lr': lr_backbone},
        {'params': model.head_parameters(), 'lr': lr_head},
    ], weight_decay=weight_decay)


def build_stage2_scheduler(optimizer, warmup_epochs, stage2_epochs):
    """Linear warmup then cosine decay over stage 2."""
    warmup_epochs = min(warmup_epochs, max(stage2_epochs - 1, 0))
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(stage2_epochs, 1))
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs - warmup_epochs),
        ],
        milestones=[warmup_epochs],
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp,
                    backbone_frozen=False, mixup_alpha=0.0, cutmix_alpha=0.0, mixup_prob=0.0):
    set_train_mode(model, backbone_frozen)
    running_loss = 0.0
    for images, metadata, targets in dataloader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        images, targets_a, targets_b, lam = maybe_mixup_cutmix(
            images, targets, mixup_alpha, cutmix_alpha, mixup_prob)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images, metadata)
            loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_targets, all_probs = [], []

    for images, metadata, targets in dataloader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets_dev = targets.to(device, non_blocking=True).unsqueeze(1)

        logits = model(images, metadata)
        val_loss += criterion(logits, targets_dev).item() * images.size(0)

        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(targets.numpy())

    val_loss /= len(dataloader.dataset)
    metrics = evaluate_predictions(np.array(all_targets), np.array(all_probs))
    return val_loss, metrics, np.array(all_targets), np.array(all_probs)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the final multi-modal melanoma model.")
    parser.add_argument('--metadata_csv', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed', 'metadata_clean.csv'))
    parser.add_argument('--img_dir', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed', 'train_512'))
    parser.add_argument('--out_dir', type=str, default=os.path.join(BASE_DIR, 'models'))

    parser.add_argument('--backbone', type=str, default='tf_efficientnet_b4_ns',
                        help="any timm model name (eva02_tiny_patch14_336 needs --image_size 336)")
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    parser.add_argument('--meta_dropout', type=float, default=0.3)
    parser.add_argument('--no_gem', action='store_true',
                        help="use the backbone's own average pooling instead of GeM pooling")
    parser.add_argument('--gem_p', type=float, default=3.0)
    parser.add_argument('--no_metadata', action='store_true',
                        help="image-only ablation: skip the tabular branch entirely")

    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'bce'])
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    parser.add_argument('--epochs', type=int, default=15, help="TOTAL epochs, stage 1 + stage 2")
    parser.add_argument('--freeze_epochs', type=int, default=3,
                        help="stage 1 length: epochs with the backbone frozen")
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help="linear LR warmup at the start of stage 2 (post-unfreeze)")
    parser.add_argument('--lr_head', type=float, default=1e-3,
                        help="LR for everything except the backbone, both stages")
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help="LR for the backbone once stage 2 unfreezes it")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help="Beta(alpha, alpha) mixup strength; 0 disables mixup")
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help="Beta(alpha, alpha) cutmix strength; 0 disables cutmix")
    parser.add_argument('--mixup_prob', type=float, default=0.5,
                        help="probability a given training batch gets mixup OR cutmix applied")

    # val picks the best epoch; test is held out.
    parser.add_argument('--val_fold', type=int, default=3)
    parser.add_argument('--test_fold', type=int, default=4)
    parser.add_argument('--no_external', action='store_true',
                        help="drop the ISIC 2019 rows from training")
    return parser


def main(args=None):
    parser = build_arg_parser()
    args = parser.parse_args(args)

    if args.freeze_epochs >= args.epochs:
        raise ValueError(f"--freeze_epochs ({args.freeze_epochs}) must be < --epochs ({args.epochs}); "
                         "stage 2 needs at least 1 epoch.")

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"--- Training final model ({args.backbone}) on {device} ---")
    print(f"  stage 1 (backbone frozen): epochs 1-{args.freeze_epochs}, lr_head={args.lr_head}")
    print(f"  stage 2 (fine-tune all):   epochs {args.freeze_epochs + 1}-{args.epochs}, "
          f"lr_backbone={args.lr_backbone}, lr_head={args.lr_head}, warmup={args.warmup_epochs}")

    if args.val_fold == args.test_fold:
        raise ValueError(
            f"val_fold and test_fold are both {args.val_fold}. They must differ, "
            "otherwise the checkpoint is selected on the same data it would be scored on."
        )

    if not os.path.exists(args.metadata_csv):
        raise FileNotFoundError(
            f"{args.metadata_csv} not found. Run the preprocessing first "
            "(src/step2_make_folds.py)."
        )
    df = pd.read_csv(args.metadata_csv)
    df, meta_cols = select_metadata_columns(df)
    use_metadata = not args.no_metadata

    if args.no_external:
        df = df[df['is_external'] == 0]

    train_df = df[~df['fold'].isin([args.val_fold, args.test_fold])].reset_index(drop=True)
    valid_df = df[df['fold'] == args.val_fold].reset_index(drop=True)

    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("train or validation split came out empty, check the fold numbers")
    if (valid_df['is_external'] == 1).any():
        raise AssertionError("external rows leaked into the validation fold")

    n_pos = int(train_df['target'].sum())
    print(f"  train: {len(train_df)} photos, {n_pos} melanoma ({n_pos / len(train_df) * 100:.2f}%)")
    print(f"  val:   {len(valid_df)} photos, {int(valid_df['target'].sum())} melanoma "
          f"({valid_df['target'].mean() * 100:.2f}%)  [fold {args.val_fold}]")
    print(f"  metadata columns: {meta_cols if use_metadata else '(disabled, image-only)'}")

    check_images_exist(train_df, args.img_dir)

    # Always pass meta_cols so the dataset keeps its 3-tuple shape; the model
    # ignores the metadata tensor when use_metadata=False.
    train_dataset = MelanomaDataset(train_df, args.img_dir, image_size=args.image_size,
                                    transform=build_augmentation(), meta_cols=meta_cols)
    valid_dataset = MelanomaDataset(valid_df, args.img_dir, image_size=args.image_size,
                                    transform=None, meta_cols=meta_cols)

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0)

    model = SkinMelanomaFinalModel(
        backbone_name=args.backbone,
        drop_rate=args.drop_rate,
        meta_dropout=args.meta_dropout,
        proj_dim=args.proj_dim,
        use_gem=not args.no_gem,
        gem_p=args.gem_p,
        num_sex=NUM_SEX_CATEGORIES,
        num_site=NUM_SITE_CATEGORIES,
        use_metadata=use_metadata,
    ).to(device)

    criterion = build_loss(args, train_df, device)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    weights_path = os.path.join(args.out_dir, f"final_{args.backbone}_fold{args.val_fold}.pth")

    best_auc = -1.0
    best_epoch = 0

    def run_validation_and_maybe_save(epoch):
        nonlocal best_auc, best_epoch
        val_loss, val_metrics, val_targets, val_probs = validate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | Val Loss: {val_loss:.4f} | "
              f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | Val PR-AUC: {val_metrics['pr_auc']:.4f}")
        if val_metrics['roc_auc'] >= best_auc:
            best_auc = val_metrics['roc_auc']
            best_epoch = epoch
            torch.save(model.state_dict(), weights_path)
            threshold, f1_at_threshold = find_best_threshold(val_targets, val_probs)
            with open(weights_path.replace('.pth', '_info.json'), 'w') as handle:
                json.dump({
                    'backbone': args.backbone,
                    'meta_cols': meta_cols if use_metadata else None,
                    'use_gem': not args.no_gem,
                    'gem_p': args.gem_p,
                    'proj_dim': args.proj_dim,
                    'freeze_epochs': args.freeze_epochs,
                    'lr_head': args.lr_head,
                    'lr_backbone': args.lr_backbone,
                    'image_size': args.image_size,
                    'loss': args.loss,
                    'val_fold': args.val_fold,
                    'test_fold': args.test_fold,
                    'epoch': epoch,
                    'val_roc_auc': float(val_metrics['roc_auc']),
                    'val_pr_auc': float(val_metrics['pr_auc']),
                    'best_threshold': float(threshold),
                    'f1_at_best_threshold': float(f1_at_threshold),
                    'used_external': not args.no_external,
                    'seed': args.seed,
                }, handle, indent=2)
            print(f"  --> saved best checkpoint (Val ROC-AUC {best_auc:.4f})")

    # --- Stage 1: backbone frozen, head-only warmup ---
    if args.freeze_epochs > 0:
        optimizer = build_stage1_optimizer(model, args.lr_head, args.weight_decay)
        for epoch in range(1, args.freeze_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, use_amp,
                backbone_frozen=True, mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha, mixup_prob=args.mixup_prob)
            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} [stage 1, backbone frozen]")
            run_validation_and_maybe_save(epoch)

    # --- Stage 2: unfreeze, differential LR, warmup + cosine ---
    stage2_epochs = args.epochs - args.freeze_epochs
    optimizer = build_stage2_optimizer(model, args.lr_backbone, args.lr_head, args.weight_decay)
    scheduler = build_stage2_scheduler(optimizer, args.warmup_epochs, stage2_epochs)

    for offset in range(stage2_epochs):
        epoch = args.freeze_epochs + offset + 1
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp,
            backbone_frozen=False, mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha, mixup_prob=args.mixup_prob)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} [stage 2, "
              f"lr_backbone={optimizer.param_groups[0]['lr']:.2e}]")
        run_validation_and_maybe_save(epoch)

    print(f"\n--- Best Val ROC-AUC: {best_auc:.4f} at epoch {best_epoch} ---")
    print(f"Saved: {weights_path}")
    return {'best_roc_auc': best_auc, 'best_epoch': best_epoch, 'weights_path': weights_path}


if __name__ == '__main__':
    main()
