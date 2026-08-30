import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)


def evaluate_predictions(y_true, y_probs, threshold=0.5):
    """
    Score a set of predictions.

    Note on pr_auc: this uses average_precision_score, not auc(recall, precision).
    Running auc() over the precision-recall curve joins the points with straight
    lines, and because a PR curve is not monotonic that interpolation reads high.
    average_precision_score is the standard unbiased estimator and is what people
    mean when they say PR-AUC.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_preds = (y_probs >= threshold).astype(int)

    pr_auc_val = average_precision_score(y_true, y_probs)

    # roc_auc is the competition metric, so report it even though it is
    # generous at this positive rate.
    if len(np.unique(y_true)) > 1:
        roc_auc_val = roc_auc_score(y_true, y_probs)
    else:
        roc_auc_val = float('nan')

    return {
        'accuracy': accuracy_score(y_true, y_preds),
        'recall': recall_score(y_true, y_preds, zero_division=0),
        'precision': precision_score(y_true, y_preds, zero_division=0),
        'f1': f1_score(y_true, y_preds, zero_division=0),
        'pr_auc': pr_auc_val,
        'roc_auc': roc_auc_val,
        'confusion_matrix': confusion_matrix(y_true, y_preds),
        'threshold': threshold,
        'positive_rate': float(np.mean(y_true)),
    }


def find_best_threshold(y_true, y_probs):
    """
    Pick the probability cutoff that maximises F1.

    A fixed 0.5 is meaningless here. BCEWithLogitsLoss with pos_weight
    deliberately pushes the outputs upward, so the model is not calibrated
    around 0.5 and scoring there makes precision look far worse than it is.

    Only ever call this on validation data. Tuning the threshold on the test
    set is the same mistake as tuning the model on it.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # precision_recall_curve returns one more precision/recall point than it
    # returns thresholds, so drop the last point before pairing them up.
    precision, recall = precision[:-1], recall[:-1]

    denominator = precision + recall
    f1_scores = np.divide(2 * precision * recall, denominator,
                          out=np.zeros_like(denominator), where=denominator > 0)

    if len(f1_scores) == 0:
        return 0.5, 0.0

    best = int(np.argmax(f1_scores))
    return float(thresholds[best]), float(f1_scores[best])


def recall_at_specificity(y_true, y_probs, target_specificity=0.95):
    """
    How many melanomas we catch while keeping false alarms at a fixed level.

    This is the number that says whether the model would be usable in practice,
    which neither ROC-AUC nor accuracy tells you on a 1.8% positive rate.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    negatives = y_probs[y_true == 0]
    positives = y_probs[y_true == 1]
    if len(negatives) == 0 or len(positives) == 0:
        return float('nan'), float('nan')

    threshold = np.quantile(negatives, target_specificity)
    recall = float(np.mean(positives >= threshold))
    return recall, float(threshold)
