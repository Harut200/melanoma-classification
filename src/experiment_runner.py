"""
Run the whole experiment programme and write results as it goes.

Designed to survive a Kaggle session dying halfway. Every finished fold is
appended to results.csv immediately, and a restart skips whatever is already
there, so you can just run it again and it picks up where it stopped.

    python src/experiment_runner.py --time_budget_h 6

On Kaggle:

    !python experiment_runner.py --data_dir /kaggle/input/melanoma-512 \
        --out_dir /kaggle/working --time_budget_h 7
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MelanomaDataset, check_images_exist
from metrics import evaluate_predictions, find_best_threshold, recall_at_specificity
from models import get_model
from train import build_augmentation, predict, set_seed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# The programme, cheapest and most informative first. If we run out of time the
# things we lose are the ones that matter least.
#
# Experiment 2 is the one that answers the question the preprocessing phase
# raised: did adding ISIC 2019 actually help, or is the model just learning
# which year a photo came from? Same folds, same seed, one variable changed.
EXPERIMENTS = [
    {'name': 'resnet34_224_ext', 'model': 'resnet34',
     'size': 224, 'external': True, 'epochs': 10},
    {'name': 'resnet34_224_noext', 'model': 'resnet34',
     'size': 224, 'external': False, 'epochs': 10},
    {'name': 'effnetb0_224_ext', 'model': 'efficientnet_b0',
     'size': 224, 'external': True, 'epochs': 10},
    {'name': 'resnet34_384_ext', 'model': 'resnet34',
     'size': 384, 'external': True, 'epochs': 8},
]

FOLDS = [0, 1, 2, 3, 4]


def train_one_fold(config, fold, args, device):
    """Train on every fold except `fold`, score on `fold`. Returns metrics + OOF."""
    set_seed(args.seed)

    folds_df = pd.read_csv(args.folds_csv)
    if not config['external']:
        folds_df = folds_df[folds_df['is_external'] == 0]

    # fold -1 is external and never equals `fold`, so it always lands in train.
    train_df = folds_df[folds_df['fold'] != fold]
    val_df = folds_df[folds_df['fold'] == fold]

    if (val_df['is_external'] == 1).any():
        raise AssertionError("external rows leaked into the validation fold")

    n_pos = int(train_df['target'].sum())
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    augmentation = build_augmentation(config['size'])
    train_ds = MelanomaDataset(train_df, args.img_dir, config['size'], augmentation)
    val_ds = MelanomaDataset(val_df, args.img_dir, config['size'], None)

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin,
                            persistent_workers=args.num_workers > 0)

    model = get_model(config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Cosine decay: high learning rate early to move fast, low at the end to
    # settle. Matters when we only have ten epochs to work with.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])

    # Mixed precision. On a GPU this is roughly a 2x speedup for free; on CPU
    # it is disabled because there is nothing to gain.
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best = {'pr_auc': -1.0}
    best_probs = None

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * images.size(0)
        scheduler.step()

        y_true, y_probs = predict(model, val_loader, device)
        m = evaluate_predictions(y_true, y_probs)

        print(f"    epoch {epoch:2d}/{config['epochs']}  "
              f"loss {running / len(train_ds):.4f}  "
              f"PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}", flush=True)

        if m['pr_auc'] >= best['pr_auc']:
            best = m
            best['epoch'] = epoch
            best_probs = y_probs
            best_targets = y_true
            if args.save_weights:
                os.makedirs(os.path.join(args.out_dir, 'weights'), exist_ok=True)
                torch.save(model.state_dict(), os.path.join(
                    args.out_dir, 'weights', f"{config['name']}_fold{fold}.pth"))

    threshold, f1_at_t = find_best_threshold(best_targets, best_probs)
    sens95, _ = recall_at_specificity(best_targets, best_probs, 0.95)

    row = {
        'experiment': config['name'],
        'model': config['model'],
        'image_size': config['size'],
        'external': config['external'],
        'fold': fold,
        'epochs': config['epochs'],
        'best_epoch': best['epoch'],
        'pr_auc': best['pr_auc'],
        'roc_auc': best['roc_auc'],
        'sens_at_95_spec': sens95,
        'best_threshold': threshold,
        'f1_at_best_threshold': f1_at_t,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'val_positives': int(val_df['target'].sum()),
        'pos_weight': float(pos_weight.item()),
    }

    oof = pd.DataFrame({
        'image_name': val_df['image_name'].values,
        'target': best_targets,
        'prob': best_probs,
        'fold': fold,
        'experiment': config['name'],
    })
    return row, oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed'))
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--folds_csv', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=os.path.join(BASE_DIR, 'reports'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--time_budget_h', type=float, default=7.0)
    parser.add_argument('--only', type=str, default=None,
                        help="comma separated experiment names to run")
    parser.add_argument('--folds', type=str, default=None,
                        help="comma separated fold numbers, default all 5")
    parser.add_argument('--save_weights', action='store_true')
    args = parser.parse_args()

    if args.img_dir is None:
        args.img_dir = os.path.join(args.data_dir, 'train_512')
    if args.folds_csv is None:
        args.folds_csv = os.path.join(args.data_dir, 'folds.csv')

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'results.csv')
    oof_path = os.path.join(args.out_dir, 'oof_predictions.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    if device.type == 'cuda':
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no GPU found. This will take days, not hours.")

    folds_df = pd.read_csv(args.folds_csv)
    check_images_exist(folds_df, args.img_dir)

    experiments = EXPERIMENTS
    if args.only:
        wanted = set(args.only.split(','))
        experiments = [e for e in EXPERIMENTS if e['name'] in wanted]
    folds = [int(f) for f in args.folds.split(',')] if args.folds else FOLDS

    # Anything already in results.csv is skipped, so a killed session resumes.
    done = set()
    if os.path.exists(results_path):
        prev = pd.read_csv(results_path)
        done = set(zip(prev['experiment'], prev['fold']))
        print(f"resuming: {len(done)} (experiment, fold) pairs already finished")

    started = time.time()
    budget = args.time_budget_h * 3600

    for config in experiments:
        for fold in folds:
            if (config['name'], fold) in done:
                continue

            elapsed = time.time() - started
            if elapsed > budget:
                print(f"\nTime budget of {args.time_budget_h}h reached. "
                      f"Stopping cleanly, {len(done)} runs saved.")
                print("Run the same command again to continue from here.")
                return

            print(f"\n=== {config['name']} | fold {fold} | "
                  f"{elapsed / 3600:.2f}h elapsed ===", flush=True)
            t0 = time.time()
            row, oof = train_one_fold(config, fold, args, device)
            row['minutes'] = (time.time() - t0) / 60

            # Append immediately, so a crash costs one fold and not the run.
            pd.DataFrame([row]).to_csv(
                results_path, mode='a', header=not os.path.exists(results_path), index=False)
            oof.to_csv(oof_path, mode='a',
                       header=not os.path.exists(oof_path), index=False)
            done.add((config['name'], fold))

            print(f"  -> PR-AUC {row['pr_auc']:.4f} | ROC-AUC {row['roc_auc']:.4f} "
                  f"| {row['minutes']:.1f} min", flush=True)

    print(f"\nAll requested runs finished in {(time.time() - started) / 3600:.2f}h")
    print(f"results: {results_path}")


if __name__ == '__main__':
    main()
