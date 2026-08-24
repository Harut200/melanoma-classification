import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MelanomaDataset, check_images_exist
from models import get_model
from metrics import evaluate_predictions, find_best_threshold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    """Same seed everywhere, so two runs of the same config are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_augmentation(image_size):
    """
    Augmentation for the training set only.

    Flips and rotations are safe here because a lesion has no natural "up".
    Brightness and contrast jitter helps because the external ISIC 2019 photos
    were taken with different cameras than our 2020 ones, and we do not want the
    model keying on that difference.

    Normalisation is NOT here on purpose. The dataset always does it.
    """
    try:
        import albumentations as A
    except ImportError:
        print("  albumentations not installed, training without augmentation")
        return None

    # Affine, not ShiftScaleRotate: albumentations 2.x deprecated the latter and
    # warns on every construction.
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-30, 30), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_targets, all_probs = [], []
    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(targets.numpy())
    return np.array(all_targets), np.array(all_probs).flatten()


def main():
    parser = argparse.ArgumentParser(description="Train a melanoma classifier.")
    parser.add_argument('--model_name', type=str, default='resnet34',
                        choices=['custom_cnn', 'resnet34', 'efficientnet_b0'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    # Two different folds. The validation fold picks the best epoch; the test
    # fold is never looked at during training, so its score is honest.
    parser.add_argument('--val_fold', type=int, default=3)
    parser.add_argument('--test_fold', type=int, default=4)
    parser.add_argument('--no_external', action='store_true',
                        help="drop the ISIC 2019 rows, for the with/without experiment")
    parser.add_argument('--img_dir', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed', 'train_512'))
    parser.add_argument('--folds_csv', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed', 'folds.csv'))
    parser.add_argument('--out_dir', type=str, default=os.path.join(BASE_DIR, 'models'))
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"--- Training {args.model_name} on {device} ---")

    if not os.path.exists(args.folds_csv):
        raise FileNotFoundError(
            f"{args.folds_csv} not found. Run the preprocessing first "
            "(src/step2_make_folds.py)."
        )
    folds_df = pd.read_csv(args.folds_csv)

    if args.val_fold == args.test_fold:
        raise ValueError(
            f"val_fold and test_fold are both {args.val_fold}. They must differ, "
            "otherwise the model is selected on the same data it is scored on and "
            "the reported number is optimistic."
        )

    if args.no_external:
        folds_df = folds_df[folds_df['is_external'] == 0]

    # fold -1 is external ISIC 2019 data. It belongs in training only, and this
    # comparison keeps it there because -1 never equals val_fold or test_fold.
    train_df = folds_df[~folds_df['fold'].isin([args.val_fold, args.test_fold])]
    val_df = folds_df[folds_df['fold'] == args.val_fold]

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("train or validation split came out empty, check the fold numbers")
    if (val_df['is_external'] == 1).any():
        raise AssertionError("external rows leaked into the validation fold")

    n_pos = int(train_df['target'].sum())
    n_neg = len(train_df) - n_pos
    n_external = int((train_df['is_external'] == 1).sum())

    print(f"  train: {len(train_df)} photos ({n_external} external), "
          f"{n_pos} melanoma ({n_pos / len(train_df) * 100:.2f}%)")
    print(f"  val:   {len(val_df)} photos, {int(val_df['target'].sum())} melanoma "
          f"({val_df['target'].mean() * 100:.2f}%)  [fold {args.val_fold}]")
    print(f"  test:  fold {args.test_fold}, untouched during training")

    check_images_exist(train_df, args.img_dir)

    augmentation = build_augmentation(args.image_size)
    train_dataset = MelanomaDataset(train_df, args.img_dir, args.image_size, augmentation)
    val_dataset = MelanomaDataset(val_df, args.img_dir, args.image_size, None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    # Computed from the data, not hardcoded. Before the external data was added
    # this ratio was about 55; with ISIC 2019 in the training pool it is about
    # 9.4, and using the old value over-weights melanoma roughly six times.
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"  pos_weight: {pos_weight.item():.2f}  (computed from the training split)")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = get_model(args.model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    weights_path = os.path.join(
        args.out_dir, f"best_{args.model_name}_fold{args.val_fold}.pth")

    best_pr_auc = 0.0
    best_epoch = 0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_targets, val_probs = predict(model, val_loader, device)
        val_metrics = evaluate_predictions(val_targets, val_probs)

        print(f"\nEpoch {epoch}/{args.epochs} - Loss: {train_loss:.4f}")
        print(f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
              f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f}")

        # >= not >: when two epochs tie on PR-AUC, keep the later one. The
        # later epoch has trained longer and is usually better calibrated, and
        # with few positives exact ties do happen.
        if val_metrics['pr_auc'] >= best_pr_auc:
            best_pr_auc = val_metrics['pr_auc']
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(model.state_dict(), weights_path)

            # The threshold is tuned on validation, never on test. Saving it
            # next to the weights means whoever evaluates later uses the same one.
            threshold, f1_at_threshold = find_best_threshold(val_targets, val_probs)
            with open(weights_path.replace('.pth', '_info.json'), 'w') as handle:
                json.dump({
                    'model_name': args.model_name,
                    'val_fold': args.val_fold,
                    'test_fold': args.test_fold,
                    'epoch': epoch,
                    'image_size': args.image_size,
                    'pos_weight': pos_weight.item(),
                    'val_pr_auc': float(val_metrics['pr_auc']),
                    'best_threshold': float(threshold),
                    'f1_at_best_threshold': float(f1_at_threshold),
                    'used_external': not args.no_external,
                    'seed': args.seed,
                }, handle, indent=2)

    if best_metrics is None:
        print("\nNo epochs ran, nothing saved.")
        return

    print(f"\n--- Best Val PR-AUC ({args.model_name}): {best_pr_auc:.4f} "
          f"at epoch {best_epoch} ---")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(best_metrics['confusion_matrix'])
    print(f"\nSaved: {weights_path}")
    print(f"For an honest score, now run: "
          f"python src/evaluate_test.py --model_name {args.model_name} "
          f"--test_fold {args.test_fold}")


if __name__ == '__main__':
    main()
