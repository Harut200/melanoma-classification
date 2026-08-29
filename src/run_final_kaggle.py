"""
Run the final multi-modal model across folds on Kaggle, with TTA and
out-of-fold predictions so the folds can be ensembled afterwards.

    python src/run_final_kaggle.py --img_dir <dir> --metadata_csv <csv> \
        --backbone tf_efficientnet_b3 --image_size 300 --folds 0,1,2,3,4

Crash safe. Every finished fold is appended to results immediately and a
restart skips what is already there, which matters because a Kaggle session can
die at any point and the whole run is measured in hours.

Kaggle allows two concurrent GPU sessions. Run folds 0,1,2 in one and 3,4 in
the other and the wall clock roughly halves.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from src.dataset import MelanomaDataset, check_images_exist
    from src.metrics import evaluate_predictions, find_best_threshold, recall_at_specificity
    from src.models.final_model import SkinMelanomaFinalModel
    from src.train_final import (META_COLS, build_augmentation, build_loss,
                                 build_stage1_optimizer, build_stage2_optimizer,
                                 build_stage2_scheduler, set_seed, train_one_epoch)
except ImportError:
    from dataset import MelanomaDataset, check_images_exist
    from metrics import evaluate_predictions, find_best_threshold, recall_at_specificity
    from models.final_model import SkinMelanomaFinalModel
    from train_final import (META_COLS, build_augmentation, build_loss,
                             build_stage1_optimizer, build_stage2_optimizer,
                             build_stage2_scheduler, set_seed, train_one_epoch)


@torch.no_grad()
def predict(model, loader, device, tta=1):
    """
    Predict, optionally averaging over flips.

    tta=1 plain, 2 adds a horizontal flip, 4 adds vertical and both. A lesion
    has no natural orientation, so flips are label preserving here and cost
    only extra forward passes.
    """
    model.eval()
    all_targets, all_probs = [], []

    for images, metadata, targets in loader:
        images = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

        views = [images]
        if tta >= 2:
            views.append(torch.flip(images, dims=[3]))
        if tta >= 4:
            views.append(torch.flip(images, dims=[2]))
            views.append(torch.flip(images, dims=[2, 3]))

        probs = torch.zeros(images.size(0), 1, device=device)
        for view in views:
            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                probs += torch.sigmoid(model(view, metadata).float())
        probs /= len(views)

        all_probs.extend(probs.squeeze(-1).cpu().numpy())
        all_targets.extend(targets.numpy())

    return np.array(all_targets), np.array(all_probs)


def run_fold(args, fold, device):
    set_seed(args.seed)

    df = pd.read_csv(args.metadata_csv)
    if args.no_external:
        df = df[df['is_external'] == 0]

    # fold -1 is external and never equals a real fold, so it always lands in
    # training and can never reach validation.
    train_df = df[~df['fold'].isin([fold, args.test_fold])].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    if (valid_df['is_external'] == 1).any():
        raise AssertionError("external rows leaked into the validation fold")
    if len(valid_df) == 0:
        raise ValueError(f"fold {fold} is empty")

    train_ds = MelanomaDataset(train_df, args.img_dir, args.image_size,
                               build_augmentation(), meta_cols=META_COLS)
    valid_ds = MelanomaDataset(valid_df, args.img_dir, args.image_size,
                               None, meta_cols=META_COLS)

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0)

    model = SkinMelanomaFinalModel(
        backbone_name=args.backbone, pretrained=True, drop_rate=args.drop_rate,
        meta_dropout=args.meta_dropout, proj_dim=args.proj_dim,
        use_metadata=not args.no_metadata,
    ).to(device)

    criterion = build_loss(args, train_df, device)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    optimizer = build_stage1_optimizer(model, args.lr_head, args.weight_decay)
    scheduler = None

    best = {'roc_auc': -1.0, 'pr_auc': -1.0}
    best_probs = best_targets = None

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        if epoch == args.freeze_epochs + 1:
            optimizer = build_stage2_optimizer(model, args.lr_backbone,
                                               args.lr_head, args.weight_decay)
            scheduler = build_stage2_scheduler(optimizer, args.warmup_epochs,
                                               args.epochs - args.freeze_epochs)

        loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                               scaler, use_amp, backbone_frozen=frozen,
                               mixup_alpha=args.mixup_alpha,
                               cutmix_alpha=args.cutmix_alpha,
                               mixup_prob=args.mixup_prob)
        if scheduler is not None:
            scheduler.step()

        # TTA only on the final epochs; it costs forward passes we do not need
        # while the model is still moving.
        tta = args.tta if epoch > args.epochs - 3 else 1
        y_true, y_prob = predict(model, valid_loader, device, tta=tta)
        m = evaluate_predictions(y_true, y_prob)

        stage = 'frozen' if frozen else 'full  '
        print(f"    epoch {epoch:2d}/{args.epochs} [{stage}] loss {loss:.4f}  "
              f"ROC-AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}"
              f"{'  (tta)' if tta > 1 else ''}", flush=True)

        # Selected on ROC-AUC to match the competition metric, but PR-AUC is
        # recorded too so this is comparable with the resnet34 baseline.
        if m['roc_auc'] >= best['roc_auc']:
            best = m
            best['epoch'] = epoch
            best_probs, best_targets = y_prob, y_true
            if args.save_weights:
                os.makedirs(os.path.join(args.out_dir, 'weights'), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(
                    args.out_dir, 'weights', f"final_{args.backbone}_fold{fold}.pth"))

    threshold, f1 = find_best_threshold(best_targets, best_probs)
    sens95, _ = recall_at_specificity(best_targets, best_probs, 0.95)

    row = {
        'experiment': f"final_{args.backbone}_{args.image_size}",
        'backbone': args.backbone, 'image_size': args.image_size,
        'external': not args.no_external, 'metadata': not args.no_metadata,
        'fold': fold, 'epochs': args.epochs, 'best_epoch': best['epoch'],
        'roc_auc': best['roc_auc'], 'pr_auc': best['pr_auc'],
        'sens_at_95_spec': sens95, 'best_threshold': threshold,
        'f1_at_best_threshold': f1, 'tta': args.tta,
        'n_train': len(train_df), 'n_val': len(valid_df),
        'val_positives': int(valid_df['target'].sum()),
    }
    oof = pd.DataFrame({
        'image_name': valid_df['image_name'].values,
        'target': best_targets, 'prob': best_probs, 'fold': fold,
        'experiment': row['experiment'],
    })
    return row, oof


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir', required=True)
    p.add_argument('--metadata_csv', required=True)
    p.add_argument('--out_dir', default='/kaggle/working/reports')
    p.add_argument('--backbone', default='tf_efficientnet_b3')
    p.add_argument('--image_size', type=int, default=300)
    p.add_argument('--folds', default='0,1,2,3,4')
    p.add_argument('--test_fold', type=int, default=-99,
                   help="held out from training everywhere; -99 means none")
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--freeze_epochs', type=int, default=2)
    p.add_argument('--warmup_epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_backbone', type=float, default=3e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--drop_rate', type=float, default=0.3)
    p.add_argument('--meta_dropout', type=float, default=0.3)
    p.add_argument('--proj_dim', type=int, default=256)
    p.add_argument('--loss', default='bce', choices=['bce', 'focal'])
    p.add_argument('--focal_alpha', type=float, default=0.25)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--mixup_alpha', type=float, default=0.4)
    p.add_argument('--cutmix_alpha', type=float, default=1.0)
    p.add_argument('--mixup_prob', type=float, default=0.5)
    p.add_argument('--tta', type=int, default=4, choices=[1, 2, 4])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--time_budget_h', type=float, default=10.0)
    p.add_argument('--no_external', action='store_true')
    p.add_argument('--no_metadata', action='store_true')
    p.add_argument('--save_weights', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise SystemExit("No GPU. This needs one.")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    if arch not in torch.cuda.get_arch_list():
        raise SystemExit(f"{name} is {arch}; this PyTorch supports "
                         f"{torch.cuda.get_arch_list()}. Use a T4.")
    print(f"device: {name} ({arch})")
    print(f"model: {args.backbone} @ {args.image_size}px, {args.epochs} epochs, tta={args.tta}")

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'final_results.csv')
    oof_path = os.path.join(args.out_dir, 'final_oof.csv')

    check_images_exist(pd.read_csv(args.metadata_csv), args.img_dir)

    done = set()
    if os.path.exists(results_path):
        prev = pd.read_csv(results_path)
        done = set(zip(prev['experiment'], prev['fold']))
        print(f"resuming: {len(done)} folds already finished")

    started = time.time()
    for fold in [int(f) for f in args.folds.split(',')]:
        exp = f"final_{args.backbone}_{args.image_size}"
        if (exp, fold) in done:
            continue
        if time.time() - started > args.time_budget_h * 3600:
            print(f"\ntime budget reached, stopping cleanly. Rerun to continue.")
            return

        print(f"\n=== {exp} | fold {fold} | {(time.time()-started)/3600:.2f}h elapsed ===",
              flush=True)
        t0 = time.time()
        row, oof = run_fold(args, fold, device)
        row['minutes'] = (time.time() - t0) / 60

        pd.DataFrame([row]).to_csv(results_path, mode='a',
                                   header=not os.path.exists(results_path), index=False)
        oof.to_csv(oof_path, mode='a',
                   header=not os.path.exists(oof_path), index=False)
        print(f"  -> ROC-AUC {row['roc_auc']:.4f} | PR-AUC {row['pr_auc']:.4f} "
              f"| {row['minutes']:.1f} min", flush=True)

    print(f"\nfinished in {(time.time()-started)/3600:.2f}h -> {results_path}")


if __name__ == '__main__':
    main()
