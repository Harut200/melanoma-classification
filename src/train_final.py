"""
Train the final multi-modal architecture: an ImageNet-pretrained vision
backbone (convnext_tiny / efficientnet_b3 / any timm model) fused with the
tabular metadata (age, sex, anatom_site_general_challenge).

Reads data/processed/metadata_clean.csv, which step2_make_folds.py already
produced with the fold assignment, the cleaned sex/site categories, and the
target all in one file -- no merge needed.

    python src/train_final.py --epochs 15 --backbone convnext_tiny --loss focal

Model selection is on validation ROC-AUC, not PR-AUC like train_baseline.py.
ROC-AUC is the actual competition metric, and it is what run_colab.py's
Google Drive checkpointing also keys off of, so the two stay consistent.
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

# Must match SEX_CATEGORIES / SITE_CATEGORIES in step2_make_folds.py. Fixed
# category counts (not "however many show up in this fold") so every fold and
# every train/val split one-hot encodes to the exact same set of columns.
NUM_SEX_CATEGORIES = 3   # female, male, unknown
NUM_SITE_CATEGORIES = 7  # 6 body sites + unknown


def set_seed(seed):
    """Same seed everywhere, so two runs of the same config are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_metadata_features(df):
    """
    Expand the label-encoded sex_enc / site_enc columns from
    step2_make_folds.py into one-hot vectors, and pair them with the already
    normalised age_norm.

    One-hot instead of the raw integer codes: site_enc=0..6 has no ordinal
    meaning (site 3 is not "more" than site 1), and feeding a category id
    straight into a linear layer tells the model there is one.

    Returns (df_with_new_columns, meta_cols) where meta_cols is the ordered
    list of column names to feed into the model -- callers pass
    len(meta_cols) as num_meta_features.
    """
    df = df.reset_index(drop=True)

    sex_oh = pd.get_dummies(
        pd.Categorical(df['sex_enc'], categories=range(NUM_SEX_CATEGORIES)),
        prefix='sex',
    ).astype(np.float32)
    site_oh = pd.get_dummies(
        pd.Categorical(df['site_enc'], categories=range(NUM_SITE_CATEGORIES)),
        prefix='site',
    ).astype(np.float32)

    df = pd.concat([df, sex_oh, site_oh], axis=1)
    meta_cols = list(sex_oh.columns) + list(site_oh.columns) + ['age_norm']
    return df, meta_cols


def build_augmentation():
    """
    Training-only augmentation. Resize happens here too, since the final
    model is often trained at a different resolution than the 512px photos
    step3_resize_images.py produced (224 for convnext_tiny/efficientnet_b3
    by default). Normalisation is NOT here -- MelanomaDataset always does
    that, so it can never be forgotten in one script and double-applied in
    another.
    """
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


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    for images, metadata, targets in dataloader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images, metadata)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Focal loss's gradient can spike on the rare hard positives early in
        # training; clipping keeps one bad batch from blowing up the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

    parser.add_argument('--backbone', type=str, default='convnext_tiny',
                        help="any timm model name, e.g. convnext_tiny, efficientnet_b3")
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    parser.add_argument('--meta_dropout', type=float, default=0.3)
    parser.add_argument('--no_residual', action='store_true',
                        help="disable the image-projection residual connection in the fusion head")

    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'bce'])
    parser.add_argument('--focal_alpha', type=float, default=0.8)
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Two different folds, same discipline as train_baseline.py: val picks the
    # best epoch, test is never looked at during training.
    parser.add_argument('--val_fold', type=int, default=3)
    parser.add_argument('--test_fold', type=int, default=4)
    parser.add_argument('--no_external', action='store_true',
                        help="drop the ISIC 2019 rows from training")
    return parser


def main(args=None):
    parser = build_arg_parser()
    args = parser.parse_args(args)

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"--- Training final model ({args.backbone}) on {device} ---")

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
    df, meta_cols = build_metadata_features(df)

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
    print(f"  test:  fold {args.test_fold}, held out, not used by this script")
    print(f"  metadata features ({len(meta_cols)}): {meta_cols}")

    check_images_exist(train_df, args.img_dir)

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
        num_meta_features=len(meta_cols),
        drop_rate=args.drop_rate,
        meta_dropout=args.meta_dropout,
        proj_dim=args.proj_dim,
        use_residual=not args.no_residual,
    ).to(device)

    criterion = build_loss(args, train_df, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    weights_path = os.path.join(args.out_dir, f"final_{args.backbone}_fold{args.val_fold}.pth")

    best_auc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_metrics, val_targets, val_probs = validate(model, valid_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
              f"Val PR-AUC: {val_metrics['pr_auc']:.4f}")

        if val_metrics['roc_auc'] >= best_auc:
            best_auc = val_metrics['roc_auc']
            best_epoch = epoch
            torch.save(model.state_dict(), weights_path)

            threshold, f1_at_threshold = find_best_threshold(val_targets, val_probs)
            with open(weights_path.replace('.pth', '_info.json'), 'w') as handle:
                json.dump({
                    'backbone': args.backbone,
                    'meta_cols': meta_cols,
                    'proj_dim': args.proj_dim,
                    'drop_rate': args.drop_rate,
                    'meta_dropout': args.meta_dropout,
                    'use_residual': not args.no_residual,
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

    print(f"\n--- Best Val ROC-AUC: {best_auc:.4f} at epoch {best_epoch} ---")
    print(f"Saved: {weights_path}")
    return {'best_roc_auc': best_auc, 'best_epoch': best_epoch, 'weights_path': weights_path}


if __name__ == '__main__':
    main()
