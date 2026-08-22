import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MelanomaDataset
from models import get_model
from metrics import evaluate_predictions


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet34',
                        choices=['custom_cnn', 'resnet34', 'efficientnet_b0'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Training {args.model_name} on {device} ---")

    # folds_df = pd.read_csv('data/processed/folds.csv')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folds_csv_path = os.path.join(BASE_DIR, 'data', 'processed', 'folds.csv')
    train_img_dir = os.path.join(BASE_DIR, 'data', 'raw', 'train')

    folds_df = pd.read_csv(folds_csv_path)

    train_df = folds_df[folds_df['fold'] != 4]
    val_df = folds_df[folds_df['fold'] == 4]

    train_dataset = MelanomaDataset(train_df, img_dir=train_img_dir)
    val_dataset = MelanomaDataset(val_df, img_dir=train_img_dir)
    # train_dataset = MelanomaDataset(train_df, img_dir='data/raw/train/')
    # val_dataset = MelanomaDataset(val_df, img_dir='data/raw/train/')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    pos_weight = torch.tensor([55.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = get_model(args.model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_pr_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_targets, val_probs = predict(model, val_loader, device)
        val_metrics = evaluate_predictions(val_targets, val_probs)

        print(f"\nEpoch {epoch}/{args.epochs} - Loss: {train_loss:.4f}")
        print(
            f"Val PR-AUC: {val_metrics['pr_auc']:.4f} | Recall: {val_metrics['recall']:.4f} | Precision: {val_metrics['precision']:.4f} | F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        if val_metrics['pr_auc'] > best_pr_auc:
            best_pr_auc = val_metrics['pr_auc']
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f"models/best_{args.model_name}.pth")

    print(f"\n--- Best Val PR-AUC ({args.model_name}): {best_pr_auc:.4f} ---")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(val_metrics['confusion_matrix'])


if __name__ == '__main__':
    main()