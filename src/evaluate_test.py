"""
Score saved weights on a fold that was never used to select them.

    python src/evaluate_test.py --weights models/ --test_fold 4

Works for both architectures. The checkpoint says which one it is, so there is
no --model_name to get wrong: a file called final_tf_efficientnet_b4_fold0.pth
is rebuilt as the multi-modal model at 300px, best_resnet34.pth as the baseline
at 224px.

WHAT THIS IS FOR
----------------
Every number in reports/ is the best epoch measured on the fold that chose that
epoch. That is normal practice for picking a checkpoint, and it is slightly
optimistic, because the epoch was selected partly on the noise in that fold.

This script removes that. It scores a fold that took no part in either training
or checkpoint selection, so the number it prints is the honest one.

It refuses to run if the checkpoint was selected on the fold you are asking it
to score. That mistake has already been made once in this repo: training
validated on fold 4 and the old evaluation script then scored fold 4, so the
"test" number was measured on the data used to pick the model.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from src.checkpoints import find_weights, load_one_model
    from src.dataset import MelanomaDataset, check_images_exist
    from src.metrics import evaluate_predictions, recall_at_specificity
    from src.train_final import META_COLS
except ImportError:
    from checkpoints import find_weights, load_one_model
    from dataset import MelanomaDataset, check_images_exist
    from metrics import evaluate_predictions, recall_at_specificity
    from train_final import META_COLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')


def load_test_split(csv_path, test_fold, needs_metadata):
    """
    Read the fold table and return only the rows in the held-out fold.

    The final model needs sex_enc, site_enc and age_norm, which live in
    metadata_clean.csv. The plain folds.csv does not have them, so this fails
    with a useful message rather than a KeyError six lines later.
    """
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"{csv_path} not found.\n"
            "Run the preprocessing first (src/step2_make_folds.py)."
        )

    df = pd.read_csv(csv_path)

    if needs_metadata:
        missing = [c for c in META_COLS if c not in df.columns]
        if missing:
            raise SystemExit(
                f"{os.path.basename(csv_path)} is missing {missing}.\n"
                "The final model has a metadata branch, so it needs\n"
                "data/processed/metadata_clean.csv, not folds.csv.\n"
                "Pass it with --metadata_csv."
            )

    test_df = df[df['fold'] == test_fold].reset_index(drop=True)
    if len(test_df) == 0:
        raise SystemExit(f"fold {test_fold} is empty in {csv_path}")

    # The external ISIC 2019 rows carry fold -1 and belong in training only.
    # If any reached the test fold the score would be measured on a different
    # dataset than the one being reported.
    if 'is_external' in test_df.columns and (test_df['is_external'] == 1).any():
        raise SystemExit(
            "External rows are in the test fold. That fold is not usable as a "
            "clean test set."
        )

    return test_df


def refuse_if_selected_on_this_fold(specs, weight_files, test_fold):
    """
    Stop if any checkpoint was chosen using the fold we are about to score.

    The fold number is in the filename for the Kaggle runner's checkpoints and
    in the sidecar json for train_final.py's, so this is knowable without
    trusting the person running the script to remember.
    """
    guilty = [
        os.path.basename(path)
        for path, spec in zip(weight_files, specs)
        if spec.get('val_fold') == test_fold
    ]
    if guilty:
        raise SystemExit(
            f"\nThese checkpoints were selected on fold {test_fold}, which is the "
            f"fold you are asking them to be scored on:\n"
            + "".join(f"  {name}\n" for name in guilty)
            + "\nThat is not a test score, it is the validation score again. "
            f"Pick a --test_fold none of the checkpoints validated on, or pass "
            f"--allow_selection_overlap if you know what you are doing and want "
            f"the number anyway."
        )


@torch.no_grad()
def predict_fold(model, spec, loader, device, tta=1):
    """
    Probabilities for one model over the whole loader.

    tta=1 plain, 2 adds a horizontal flip, 4 adds vertical and both. A lesion
    has no natural orientation, so flips are label preserving here.
    """
    all_targets, all_probs = [], []

    for batch in loader:
        if len(batch) == 3:
            images, metadata, targets = batch
            metadata = metadata.to(device, non_blocking=True)
        else:
            images, targets = batch
            metadata = None
        images = images.to(device, non_blocking=True)

        views = [images]
        if tta >= 2:
            views.append(torch.flip(images, dims=[3]))
        if tta >= 4:
            views.append(torch.flip(images, dims=[2]))
            views.append(torch.flip(images, dims=[2, 3]))

        probs = torch.zeros(images.size(0), 1, device=device)
        for view in views:
            logits = model(view, metadata) if spec['kind'] == 'final' else model(view)
            probs += torch.sigmoid(logits.float())
        probs /= len(views)

        all_probs.extend(probs.squeeze(-1).cpu().numpy())
        all_targets.extend(targets.numpy())

    return np.array(all_targets), np.array(all_probs)


def print_report(title, y_true, y_prob, threshold, n_models):
    metrics = evaluate_predictions(y_true, y_prob, threshold=threshold)
    sens_95, _ = recall_at_specificity(y_true, y_prob, 0.95)

    print()
    print("=" * 62)
    print(f" {title}")
    print("=" * 62)
    print(f"photos      : {len(y_true)}  ({int(y_true.sum())} melanoma, "
          f"{metrics['positive_rate'] * 100:.2f}%)")
    if n_models > 1:
        print(f"models      : {n_models}, probabilities averaged")
    print(f"ROC-AUC     : {metrics['roc_auc']:.4f}   (competition metric)")
    print(f"PR-AUC      : {metrics['pr_auc']:.4f}   (random guessing scores "
          f"{metrics['positive_rate']:.4f})")
    print(f"Sens@95Spec : {sens_95:.4f}   melanomas caught at a 5% false alarm rate")
    print(f"Recall      : {metrics['recall']:.4f}   at threshold {threshold:.4f}")
    print(f"Precision   : {metrics['precision']:.4f}")
    print(f"F1          : {metrics['f1']:.4f}")
    print(f"Accuracy    : {metrics['accuracy']:.4f}   (ignore this, a model that "
          f"says 'no' every time scores 98%)")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics['confusion_matrix'])
    return metrics, sens_95


def main():
    parser = argparse.ArgumentParser(
        description="Score saved weights on a held-out fold.")
    parser.add_argument('--weights', default=os.path.join(BASE_DIR, 'models'),
                        help="a .pth file, a folder of them, or a glob "
                             "(default: models/)")
    parser.add_argument('--test_fold', type=int, default=4,
                        help="the fold to score, which no checkpoint may have "
                             "been selected on")
    parser.add_argument('--metadata_csv', default=None,
                        help="defaults to metadata_clean.csv for the final "
                             "model, folds.csv for a baseline")
    parser.add_argument('--img_dir',
                        default=os.path.join(BASE_DIR, 'data', 'processed', 'train_512'))
    parser.add_argument('--backbone', default=None,
                        help="override the architecture if it cannot be read "
                             "from the filename or sidecar json")
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=None,
                        help="default: the threshold tuned on validation and "
                             "saved beside the weights, else 0.5. Never tune "
                             "this on the test fold.")
    parser.add_argument('--tta', type=int, default=1, choices=[1, 2, 4])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default=None, choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--per_model', action='store_true',
                        help="also print each checkpoint on its own, not just "
                             "the ensemble")
    parser.add_argument('--allow_selection_overlap', action='store_true',
                        help="score even if a checkpoint was selected on this "
                             "fold. The result is not a test score.")
    parser.add_argument('--out_csv', default=None,
                        help="write the per-photo probabilities here")
    args = parser.parse_args()

    device = pick_device(args.device)
    weight_files = find_weights(args.weights)

    print(f"loading {len(weight_files)} checkpoint(s) on {device.type} ...")
    models, specs = [], []
    for path in weight_files:
        model, spec = load_one_model(path, device, args.backbone, args.image_size)
        models.append(model)
        specs.append(spec)
        fold_note = ("selected on fold %s" % spec['val_fold']
                     if spec['val_fold'] is not None else "fold unknown")
        print(f"  {os.path.basename(path)}  ->  {spec['backbone']} "
              f"@{spec['image_size']}px  ({spec['kind']}, {fold_note})")

    kinds = {spec['kind'] for spec in specs}
    if len(kinds) > 1:
        raise SystemExit(
            f"Mixed architectures in {args.weights} ({sorted(kinds)}). "
            "They cannot be averaged. Point --weights at one set."
        )
    kind = kinds.pop()

    input_sizes = {spec['image_size'] for spec in specs}
    if len(input_sizes) > 1:
        raise SystemExit(
            f"The checkpoints want different input sizes ({sorted(input_sizes)}), "
            "so they cannot be averaged."
        )
    image_size = input_sizes.pop()

    if not args.allow_selection_overlap:
        refuse_if_selected_on_this_fold(specs, weight_files, args.test_fold)

    needs_metadata = kind == 'final' and any(spec['use_metadata'] for spec in specs)
    csv_path = args.metadata_csv or os.path.join(
        BASE_DIR, 'data', 'processed',
        'metadata_clean.csv' if needs_metadata else 'folds.csv')

    test_df = load_test_split(csv_path, args.test_fold, needs_metadata)
    check_images_exist(test_df, args.img_dir)

    dataset = MelanomaDataset(test_df, args.img_dir, image_size, transform=None,
                              meta_cols=META_COLS if needs_metadata else None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == 'cuda')

    # The threshold was tuned on validation during training and saved beside the
    # weights. Re-tuning it here on the test fold would invalidate the score.
    saved = [spec['threshold'] for spec in specs if spec['threshold'] is not None]
    if args.threshold is not None:
        threshold = args.threshold
        print(f"\nthreshold {threshold:.4f} (passed on the command line)")
    elif saved:
        threshold = float(np.mean(saved))
        print(f"\nthreshold {threshold:.4f} (tuned on validation, from "
              f"{len(saved)} sidecar json file(s))")
    else:
        threshold = 0.5
        print("\nthreshold 0.5000 (no sidecar json found; this is not a tuned "
              "value and precision will look worse than it is)")

    all_probs, y_true = [], None
    for model, spec, path in zip(models, specs, weight_files):
        targets, probs = predict_fold(model, spec, loader, device, tta=args.tta)
        y_true = targets if y_true is None else y_true
        all_probs.append(probs)
        if args.per_model and len(models) > 1:
            print_report(f"{os.path.basename(path)}  -  TEST FOLD {args.test_fold}",
                         targets, probs, threshold, 1)

    y_prob = np.mean(all_probs, axis=0)
    title = (f"{specs[0]['backbone'].upper()} @{image_size}px  -  "
             f"TEST FOLD {args.test_fold}")
    if args.tta > 1:
        title += f"  ({args.tta}x TTA)"
    print_report(title, y_true, y_prob, threshold, len(models))

    if args.out_csv:
        pd.DataFrame({
            'image_name': test_df['image_name'].values,
            'target': y_true,
            'prob': y_prob,
            'fold': args.test_fold,
        }).to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == '__main__':
    main()
