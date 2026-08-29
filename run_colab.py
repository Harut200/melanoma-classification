"""
Colab/Kaggle entry point for training the final model. Wraps
src/train_final.py and checkpoints the best model to Google Drive (or any
--checkpoint_path) so a dropped runtime doesn't lose the run.

    !python run_colab.py \\
        --img_dir <train_512> --csv_path <metadata_clean.csv> \\
        --backbone tf_efficientnet_b4_ns --epochs 15 --freeze_epochs 3
"""

import sys

sys.path.append('.')  # so `from src...` works regardless of cwd

import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import MelanomaDataset, check_images_exist
from src.losses.focal_loss import BinaryFocalLoss
from src.metrics import find_best_threshold
from src.models.final_model import SkinMelanomaFinalModel
from src.train_final import (
    META_COLS,
    NUM_SEX_CATEGORIES,
    NUM_SITE_CATEGORIES,
    build_augmentation,
    build_stage1_optimizer,
    build_stage2_optimizer,
    build_stage2_scheduler,
    select_metadata_columns,
    set_seed,
    train_one_epoch,
    validate,
)

DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/melanoma_checkpoints'
DRIVE_CHECKPOINT_PATH = os.path.join(DRIVE_CHECKPOINT_DIR, 'best_model.pth')


def parse_args():
    parser = argparse.ArgumentParser(description="Train the final model in Colab.")
    parser.add_argument('--img_dir', type=str, required=True,
                        help="folder of resized training jpgs")
    parser.add_argument('--csv_path', type=str, default='./folds.csv',
                        help="metadata_clean.csv from step2_make_folds.py (needs the "
                             "sex/site/age columns, not the plain folds.csv)")

    parser.add_argument('--backbone', type=str, default='tf_efficientnet_b4_ns',
                        help="any timm model name")
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    parser.add_argument('--meta_dropout', type=float, default=0.3)
    parser.add_argument('--no_gem', action='store_true',
                        help="use the backbone's own average pooling instead of GeM pooling")
    parser.add_argument('--gem_p', type=float, default=3.0)
    parser.add_argument('--no_metadata', action='store_true',
                        help="image-only ablation: skip the tabular branch entirely")

    parser.add_argument('--epochs', type=int, default=15, help="TOTAL epochs, stage 1 + stage 2")
    parser.add_argument('--freeze_epochs', type=int, default=3,
                        help="stage 1 length: epochs with the backbone frozen")
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help="linear LR warmup at the start of stage 2 (post-unfreeze)")
    parser.add_argument('--lr_head', type=float, default=1e-3,
                        help="LR for everything except the backbone, both stages")
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help="LR for the backbone once stage 2 unfreezes it")
    parser.add_argument('--lr', type=float, default=None,
                        help="deprecated alias for --lr_head, kept for older commands; "
                             "--lr_head takes precedence if both are passed")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_fold', type=int, default=3)
    parser.add_argument('--test_fold', type=int, default=4)
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'bce'])
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--mixup_prob', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_path', type=str, default=DRIVE_CHECKPOINT_PATH,
                        help="where to save the best-so-far weights on every improvement")
    args = parser.parse_args()
    if args.lr is not None:
        args.lr_head = args.lr
    return args


def build_loss(args, train_df, device):
    if args.loss == 'focal':
        return BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    n_pos = int(train_df['target'].sum())
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def main():
    args = parse_args()
    if args.freeze_epochs >= args.epochs:
        raise ValueError(f"--freeze_epochs ({args.freeze_epochs}) must be < --epochs ({args.epochs}); "
                         "stage 2 needs at least 1 epoch.")
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > GPU.")
    print(f"--- Training final model ({args.backbone}) on {device} ---")
    print(f"  stage 1 (backbone frozen): epochs 1-{args.freeze_epochs}, lr_head={args.lr_head}")
    print(f"  stage 2 (fine-tune all):   epochs {args.freeze_epochs + 1}-{args.epochs}, "
          f"lr_backbone={args.lr_backbone}, lr_head={args.lr_head}, warmup={args.warmup_epochs}")

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir.startswith('/content/drive') and not os.path.isdir('/content/drive'):
        raise RuntimeError(
            "Google Drive is not mounted at /content/drive. Run this first:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')"
        )
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"{args.csv_path} not found. Pass --csv_path pointing at "
                                "metadata_clean.csv from src/step2_make_folds.py.")
    df = pd.read_csv(args.csv_path)

    required_cols = set(META_COLS) | {'fold', 'target'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{args.csv_path} is missing columns {sorted(missing)}.\n"
            "This looks like the plain folds.csv, not metadata_clean.csv -- folds.csv only "
            "carries image_name/patient_id/target/fold/is_external, without the sex/site/age "
            "metadata this model needs. Re-run with --csv_path pointing at metadata_clean.csv."
        )

    df, meta_cols = select_metadata_columns(df)
    use_metadata = not args.no_metadata

    if args.val_fold == args.test_fold:
        raise ValueError("--val_fold and --test_fold must differ.")

    train_df = df[~df['fold'].isin([args.val_fold, args.test_fold])].reset_index(drop=True)
    valid_df = df[df['fold'] == args.val_fold].reset_index(drop=True)
    if len(train_df) == 0 or len(valid_df) == 0:
        raise ValueError("train or validation split came out empty, check --val_fold/--test_fold")

    print(f"  train: {len(train_df)} photos, {int(train_df['target'].sum())} melanoma")
    print(f"  val:   {len(valid_df)} photos, {int(valid_df['target'].sum())} melanoma  [fold {args.val_fold}]")
    print(f"  checkpoint (best Val ROC-AUC) -> {args.checkpoint_path}")

    check_images_exist(train_df, args.img_dir)

    # Always pass meta_cols so the dataset keeps its 3-tuple shape; the model
    # ignores the metadata tensor when use_metadata=False.
    train_dataset = MelanomaDataset(train_df, args.img_dir, image_size=args.image_size,
                                    transform=build_augmentation(), meta_cols=meta_cols)
    valid_dataset = MelanomaDataset(valid_df, args.img_dir, image_size=args.image_size,
                                    transform=None, meta_cols=meta_cols)

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

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

    best_auc = -1.0

    def run_validation_and_maybe_save(epoch):
        nonlocal best_auc
        val_loss, val_metrics, val_targets, val_probs = validate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | Val Loss: {val_loss:.4f} | "
              f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        # Save on every ROC-AUC improvement so a dropped runtime keeps the best.
        if val_metrics['roc_auc'] >= best_auc:
            best_auc = val_metrics['roc_auc']
            torch.save(model.state_dict(), args.checkpoint_path)
            threshold, _ = find_best_threshold(val_targets, val_probs)
            info_path = args.checkpoint_path.replace('.pth', '_info.json')
            with open(info_path, 'w') as handle:
                json.dump({
                    'backbone': args.backbone,
                    'meta_cols': meta_cols if use_metadata else None,
                    'use_gem': not args.no_gem,
                    'gem_p': args.gem_p,
                    'freeze_epochs': args.freeze_epochs,
                    'lr_head': args.lr_head,
                    'lr_backbone': args.lr_backbone,
                    'image_size': args.image_size,
                    'val_fold': args.val_fold,
                    'test_fold': args.test_fold,
                    'epoch': epoch,
                    'val_roc_auc': float(best_auc),
                    'best_threshold': float(threshold),
                }, handle, indent=2)
            print(f"  --> saved checkpoint to Drive (Val ROC-AUC {best_auc:.4f})")

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

    print(f"\n--- Best Val ROC-AUC: {best_auc:.4f} ---")
    print(f"Checkpoint: {args.checkpoint_path}")


if __name__ == '__main__':
    main()
