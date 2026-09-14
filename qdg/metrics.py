"""Macro AUROC is the primary metric; per-class AUROC is always reported (方案 §9)."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from .data import CLASSES


def validation_thresholds(target, probability):
    """Per-class F1 thresholds chosen exclusively on validation predictions."""
    thresholds = np.full(len(CLASSES), 0.5)
    for c in range(len(CLASSES)):
        if np.unique(target[:, c]).size < 2:
            continue
        precision, recall, candidates = precision_recall_curve(target[:, c], probability[:, c])
        score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        thresholds[c] = candidates[int(np.argmax(score))]
    return thresholds


def multilabel_metrics(target, probability, thresholds=None):
    target, probability = np.asarray(target), np.asarray(probability)
    if target.shape != probability.shape or target.ndim != 2 or target.shape[1] != len(CLASSES):
        raise ValueError("Expected matching [records, 5] labels/probabilities")
    if not len(target) or not np.isfinite(probability).all():
        raise ValueError("Empty or nonfinite evaluation predictions")
    thresholds = np.full(len(CLASSES), 0.5) if thresholds is None else np.asarray(thresholds)
    predicted = probability >= thresholds[None, :]
    per_class = {}
    for c, name in enumerate(CLASSES):
        valid = np.unique(target[:, c]).size == 2
        per_class[name] = {
            "auroc": float(roc_auc_score(target[:, c], probability[:, c])) if valid else None,
            "auprc": float(average_precision_score(target[:, c], probability[:, c]))
            if valid
            else None,
            "f1": float(f1_score(target[:, c], predicted[:, c], zero_division=0)),
            "positives": int(target[:, c].sum()),
            "threshold": float(thresholds[c]),
        }
    result = {
        "records": len(target),
        "per_class": per_class,
        "valid_auc_classes": sum(item["auroc"] is not None for item in per_class.values()),
    }
    for metric in ("auroc", "auprc", "f1"):
        values = [item[metric] for item in per_class.values() if item[metric] is not None]
        result[f"macro_{metric}"] = float(np.mean(values)) if values else None
    return result
