"""
Standalone entry point for training the final multi-modal model in Google
Colab (or any single-script environment where cloning the repo and running
`python -m src.train_final` isn't convenient).

Typical Colab usage:

    from google.colab import drive
    drive.mount('/content/drive')

    !python run_colab.py \\
        --img_dir /content/drive/MyDrive/melanoma/train_512 \\
        --csv_path /content/drive/MyDrive/melanoma/metadata_clean.csv \\
        --epochs 15 --batch_size 32 --lr 3e-4

This script is a thin wrapper around src/train_final.py: it reuses that
module's dataset/model/loss building blocks so there is exactly one place
that defines how the final model is trained, and adds the one thing that is
genuinely Colab-specific -- checkpointing the best-so-far weights straight to
Google Drive so a disconnected runtime doesn't lose the run.
"""

import sys

sys.path.append('.')  # repo root, so `from src...` works regardless of cwd

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
    build_augmentation,
    build_metadata_features,
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
                        help="CSV with the fold assignment, sex/site encoding, and target. "
                             "Despite the default filename, this needs to be "
                             "metadata_clean.csv from src/step2_make_folds.py -- the plain "
                             "folds.csv it also writes only has "
                             "image_name/patient_id/target/fold/is_external, no metadata "
                             "columns, and will fail the check below. Point this at your "
                             "actual metadata_clean.csv (or a copy/symlink named folds.csv).")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)

    parser.add_argument('--backbone', type=str, default='convnext_tiny')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_fold', type=int, default=3)
    parser.add_argument('--test_fold', type=int, default=4)
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'bce'])
    parser.add_argument('--focal_alpha', type=float, default=0.8)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_path', type=str, default=DRIVE_CHECKPOINT_PATH,
                        help="where to save the best-so-far weights on every improvement")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > GPU.")
    print(f"--- Training final model ({args.backbone}) on {device} ---")

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

    required_cols = {'sex_enc', 'site_enc', 'age_norm', 'fold', 'target'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{args.csv_path} is missing columns {sorted(missing)}.\n"
            "This looks like the plain folds.csv, not metadata_clean.csv -- folds.csv only "
            "carries image_name/patient_id/target/fold/is_external, without the sex/site/age "
            "metadata this model needs. Re-run with --csv_path pointing at metadata_clean.csv."
        )

    df, meta_cols = build_metadata_features(df)

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
        backbone_name=args.backbone, num_meta_features=len(meta_cols),
    ).to(device)

    if args.loss == 'focal':
        criterion = BinaryFocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        n_pos = int(train_df['target'].sum())
        n_neg = len(train_df) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_metrics, val_targets, val_probs = validate(model, valid_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        # Checkpoint to Drive on every Val ROC-AUC improvement, so a runtime
        # that disconnects mid-run has already saved the best weights seen so
        # far -- not just whatever the last completed epoch happened to be.
        if val_metrics['roc_auc'] >= best_auc:
            best_auc = val_metrics['roc_auc']
            torch.save(model.state_dict(), args.checkpoint_path)

            threshold, _ = find_best_threshold(val_targets, val_probs)
            info_path = args.checkpoint_path.replace('.pth', '_info.json')
            with open(info_path, 'w') as handle:
                json.dump({
                    'backbone': args.backbone,
                    'meta_cols': meta_cols,
                    'image_size': args.image_size,
                    'val_fold': args.val_fold,
                    'test_fold': args.test_fold,
                    'epoch': epoch,
                    'val_roc_auc': float(best_auc),
                    'best_threshold': float(threshold),
                }, handle, indent=2)
            print(f"  --> saved checkpoint to Drive (Val ROC-AUC {best_auc:.4f})")

    print(f"\n--- Best Val ROC-AUC: {best_auc:.4f} ---")
    print(f"Checkpoint: {args.checkpoint_path}")


if __name__ == '__main__':
    main()
