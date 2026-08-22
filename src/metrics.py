import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, precision_recall_curve, auc,
    confusion_matrix
)


def evaluate_predictions(y_true, y_probs, threshold=0.5):
    y_preds = (y_probs >= threshold).astype(int)

    precision_array, recall_array, _ = precision_recall_curve(y_true, y_probs)
    pr_auc_val = auc(recall_array, precision_array)

    acc = accuracy_score(y_true, y_preds)
    rec = recall_score(y_true, y_preds, zero_division=0)
    prec = precision_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    cm = confusion_matrix(y_true, y_preds)

    return {
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'f1': f1,
        'pr_auc': pr_auc_val,
        'confusion_matrix': cm
    }