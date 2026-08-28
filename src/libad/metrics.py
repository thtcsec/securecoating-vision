"""Academic LIBAD metrics plus SecureCoating-Vision industrial gate metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _binary_labels(labels: Sequence[int]) -> np.ndarray:
    array = np.asarray(labels).reshape(-1)
    return (array > 0).astype(np.int32)


def auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = s[y == 1]
    negatives = s[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    greater = np.sum(positives[:, None] > negatives[None, :])
    equal = np.sum(positives[:, None] == negatives[None, :])
    return float((greater + 0.5 * equal) / (len(positives) * len(negatives)))


def aupr(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(np.unique(y)) < 2:
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(int(y.sum()), 1)
    precision = tp / np.clip(tp + fp, 1, None)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def f1_max(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(np.unique(y)) < 2:
        return None
    best = 0.0
    for threshold in np.unique(s):
        pred = s >= threshold
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum(~pred & (y == 1)))
        denom = 2 * tp + fp + fn
        if denom <= 0:
            continue
        best = max(best, 2 * tp / denom)
    return float(best)


def fpr_at_tpr(
    labels: Sequence[int],
    scores: Sequence[float],
    target_tpr: float = 0.95,
) -> Optional[float]:
    y = _binary_labels(labels)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = 0
    fp = 0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        if tp / n_pos >= target_tpr:
            return float(fp / n_neg)
    return 1.0


def academic_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "auroc": auroc(labels, scores),
        "aupr": aupr(labels, scores),
        "f1_max": f1_max(labels, scores),
        "fpr95": fpr_at_tpr(labels, scores, target_tpr=0.95),
    }


def summarize_splits(records: Iterable[Dict[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    collected: Dict[str, List[float]] = {}
    for record in records:
        for key, value in record.items():
            if value is None:
                continue
            collected.setdefault(key, []).append(float(value))
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for key, values in collected.items():
        array = np.asarray(values, dtype=np.float64)
        out[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "n": int(len(array)),
        }
    return out


def industrial_gate_metrics(
    labels: Sequence[int],
    decisions: Sequence[str],
) -> Dict[str, Optional[float]]:
    """SecureCoating-Vision operational metrics.

    Automatic Decision Coverage = (N_PASS + N_REJECT) / N
    HOLD Rate = N_HOLD / N
    Escape Rate = (anomalies incorrectly PASS) / (total anomalies)
    Selective Risk = (errors among automatic decisions) / (N_PASS + N_REJECT)
    """
    y = _binary_labels(labels)
    gate = np.asarray([str(item).upper() for item in decisions])
    if len(y) != len(gate):
        raise ValueError("labels and decisions must have the same length")
    n_total = len(gate)
    n_pass = int(np.sum(gate == "PASS"))
    n_reject = int(np.sum(gate == "REJECT"))
    n_hold = int(np.sum(gate == "HOLD"))
    n_auto = n_pass + n_reject
    n_anom = int(np.sum(y == 1))
    escapes = int(np.sum((y == 1) & (gate == "PASS")))
    auto_errors = int(
        np.sum(((gate == "PASS") & (y == 1)) | ((gate == "REJECT") & (y == 0)))
    )
    return {
        "n": n_total,
        "n_pass": n_pass,
        "n_reject": n_reject,
        "n_hold": n_hold,
        "automatic_decision_coverage": (n_auto / n_total) if n_total else None,
        "hold_rate": (n_hold / n_total) if n_total else None,
        "escape_rate": (escapes / n_anom) if n_anom else 0.0,
        "selective_risk": (auto_errors / n_auto) if n_auto else None,
        "n_escape": escapes,
        "n_auto_errors": auto_errors,
    }
