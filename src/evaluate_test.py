import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import MelanomaDataset
from models import get_model
from metrics import evaluate_predictions


def run_test_evaluation(model_name, weight_path):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    folds_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'folds.csv'))
    test_df = folds_df[folds_df['fold'] == 4].reset_index(drop=True)

    test_dataset = MelanomaDataset(test_df, img_dir=os.path.join(BASE_DIR, 'data', 'raw', 'train'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = get_model(model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    all_targets, all_probs = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())

    metrics = evaluate_predictions(np.array(all_targets), np.array(all_probs).flatten())

    print(f"\n================ {model_name.upper()} TEST RESULTS ================")
    print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(metrics['confusion_matrix'])


# if __name__ == '__main__':
#     run_test_evaluation('custom_cnn', 'models/best_custom_cnn.pth')
#     run_test_evaluation('resnet34', 'models/best_resnet34.pth')
#     run_test_evaluation('efficientnet_b0', 'models/best_efficientnet_b0.pth')

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cnn_weights = os.path.join(BASE_DIR, 'models', 'best_custom_cnn.pth')
    resnet_weights = os.path.join(BASE_DIR, 'models', 'best_resnet34.pth')
    effnet_weights = os.path.join(BASE_DIR, 'models', 'best_efficientnet_b0.pth')

    if os.path.exists(cnn_weights):
        run_test_evaluation('custom_cnn', cnn_weights)
    else:
        print(f"Skipping custom_cnn: {cnn_weights} not found. Run train.py first!")

    if os.path.exists(resnet_weights):
        run_test_evaluation('resnet34', resnet_weights)
    else:
        print(f"Skipping resnet34: {resnet_weights} not found. Run train.py first!")

    if os.path.exists(effnet_weights):
        run_test_evaluation('efficientnet_b0', effnet_weights)
    else:
        print(f"Skipping efficientnet_b0: {effnet_weights} not found. Run train.py first!")