"""Metrics for the supervised protocol: the qdg per-class set plus the micro averages."""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from qdg.metrics import multilabel_metrics


def full_metrics(target, probability, thresholds=None):
    """Macro/micro AUROC and F1, macro AUPRC and every per-label value.

    F1 uses ``thresholds`` when given and 0.5 otherwise; AUROC is threshold-free.
    """
    result = multilabel_metrics(target, probability, thresholds)
    cut = 0.5 if thresholds is None else np.asarray(thresholds)[None, :]
    predicted = np.asarray(probability) >= cut
    result["micro_auroc"] = float(roc_auc_score(target, probability, average="micro"))
    result["micro_f1"] = float(f1_score(target, predicted, average="micro", zero_division=0))
    return result
