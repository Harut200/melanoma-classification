import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MelanomaDataset, check_images_exist
from metrics import evaluate_predictions, recall_at_specificity
from models.baseline_models import get_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_test_evaluation(model_name, weight_path, folds_csv, img_dir,
                        test_fold, image_size, batch_size, num_workers):
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')

    folds_df = pd.read_csv(folds_csv)

    # The test fold must be a fold the training run never selected on. The
    # previous version evaluated on fold 4 while train.py also validated on
    # fold 4, so the "test" score was measured on the data used to pick the
    # best epoch.
    test_df = folds_df[folds_df['fold'] == test_fold].reset_index(drop=True)
    if len(test_df) == 0:
        raise ValueError(f"fold {test_fold} is empty in {folds_csv}")
    if (test_df['is_external'] == 1).any():
        raise AssertionError("external rows are in the test fold, that fold is not usable")

    # The threshold was tuned on validation during training and saved next to
    # the weights. Re-tuning it here on the test set would invalidate the score.
    info_path = weight_path.replace('.pth', '_info.json')
    threshold = 0.5
    if os.path.exists(info_path):
        with open(info_path) as handle:
            info = json.load(handle)
        threshold = info.get('best_threshold', 0.5)
        if info.get('val_fold') == test_fold:
            raise AssertionError(
                f"This checkpoint was selected on fold {test_fold}, which is the "
                "fold you are now testing on. Retrain with a different --val_fold."
            )
        print(f"  using threshold {threshold:.4f} tuned on validation fold "
              f"{info.get('val_fold')}")
    else:
        print(f"  no {os.path.basename(info_path)} found, falling back to threshold 0.5")

    check_images_exist(test_df, img_dir)

    test_dataset = MelanomaDataset(test_df, img_dir, image_size, None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    model = get_model(model_name, pretrained=False).to(device)
    # weights_only=True refuses to unpickle arbitrary objects out of the file.
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    all_targets, all_probs = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images.to(device))
            all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(targets.numpy())

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs).flatten()

    metrics = evaluate_predictions(y_true, y_probs, threshold=threshold)
    sens_95, _ = recall_at_specificity(y_true, y_probs, 0.95)

    print(f"\n================ {model_name.upper()} - TEST FOLD {test_fold} ================")
    print(f"photos    : {len(y_true)}  ({int(y_true.sum())} melanoma, "
          f"{metrics['positive_rate'] * 100:.2f}%)")
    print(f"PR-AUC    : {metrics['pr_auc']:.4f}   (random baseline is "
          f"{metrics['positive_rate']:.4f})")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}   (competition metric)")
    print(f"Recall    : {metrics['recall']:.4f}   at threshold {threshold:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"Sens@95Spec: {sens_95:.4f}   (melanomas caught at 5% false alarms)")
    print(f"Accuracy  : {metrics['accuracy']:.4f}   (ignore this, it is 98% for a "
          f"model that says 'no' every time)")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(metrics['confusion_matrix'])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Score a trained model on a held-out fold.")
    parser.add_argument('--model_name', type=str, default=None,
                        help="one model, or leave out to score every checkpoint found")
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--test_fold', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_dir', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed', 'train_512'))
    parser.add_argument('--folds_csv', type=str,
                        default=os.path.join(BASE_DIR, 'data', 'processed', 'folds.csv'))
    parser.add_argument('--models_dir', type=str, default=os.path.join(BASE_DIR, 'models'))
    args = parser.parse_args()

    if args.model_name and args.weights:
        targets = [(args.model_name, args.weights)]
    elif args.model_name:
        pattern = f"best_{args.model_name}_fold"
        found = sorted(f for f in os.listdir(args.models_dir)
                       if f.startswith(pattern) and f.endswith('.pth'))
        targets = [(args.model_name, os.path.join(args.models_dir, f)) for f in found]
    else:
        targets = []
        if os.path.isdir(args.models_dir):
            for f in sorted(os.listdir(args.models_dir)):
                if f.startswith('best_') and f.endswith('.pth'):
                    name = f[len('best_'):].rsplit('_fold', 1)[0]
                    targets.append((name, os.path.join(args.models_dir, f)))

    if not targets:
        print(f"No checkpoints found in {args.models_dir}. Run src/train.py first.")
        return

    for model_name, weight_path in targets:
        if not os.path.exists(weight_path):
            print(f"Skipping {model_name}: {weight_path} not found")
            continue
        run_test_evaluation(model_name, weight_path, args.folds_csv, args.img_dir,
                            args.test_fold, args.image_size, args.batch_size,
                            args.num_workers)


if __name__ == '__main__':
    main()
