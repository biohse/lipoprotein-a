from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)

MetricFunction = Callable[[np.ndarray, np.ndarray], dict[str, float]]


def model_score(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    return np.asarray(model.predict(features))


def select_threshold(y_true: np.ndarray, score: np.ndarray, min_specificity: float = 0.9) -> float:
    false_positive_rate, sensitivity, thresholds = roc_curve(y_true, score, drop_intermediate=False)
    specificity = 1 - false_positive_rate
    eligible = specificity >= min_specificity
    order = np.lexsort((thresholds[eligible], specificity[eligible], sensitivity[eligible]))
    return float(thresholds[eligible][order[-1]])


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    total = len(y_true)
    prevalence = float(np.mean(y_true))
    ppv = _divide(tp, tp + fp)
    overall_nnt = _divide(1, prevalence)
    model_nnt = _divide(1, ppv)

    return {
        "auroc": float(roc_auc_score(y_true, score)) if np.unique(y_true).size == 2 else float("nan"),
        "auprc": float(average_precision_score(y_true, score)) if np.any(y_true) else float("nan"),
        "prevalence": prevalence,
        "sensitivity": _divide(tp, tp + fn),
        "specificity": _divide(tn, tn + fp),
        "ppv": ppv,
        "npv": _divide(tn, tn + fn),
        "accuracy": _divide(tp + tn, total),
        "f1": _divide(2 * tp, 2 * tp + fp + fn),
        "testing_proportion": _divide(tp + fp, total),
        "overall_nnt": overall_nnt,
        "model_nnt": model_nnt,
        "nnt_relative_reduction": _divide(overall_nnt - model_nnt, overall_nnt),
        "tests_per_1000": _divide(tp + fp, total) * 1000,
        "tp_per_1000": _divide(tp, total) * 1000,
        "fp_per_1000": _divide(fp, total) * 1000,
        "fn_per_1000": _divide(fn, total) * 1000,
        "tn_per_1000": _divide(tn, total) * 1000,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def regression_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, score),
        "rmse": mean_squared_error(y_true, score) ** 0.5,
        "spearman": float(spearmanr(y_true, score)[0]),  # pyright: ignore[reportArgumentType]
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    metric: MetricFunction,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_bootstrap):
        indices = random.integers(0, len(y_true), len(y_true))
        rows.append({"replicate": replicate, **metric(y_true[indices], score[indices])})
    return pd.DataFrame(rows)


def metric_summary(point: dict[str, float], bootstrap: pd.DataFrame) -> pd.DataFrame:
    counts = {"tp", "fp", "fn", "tn"}
    rows = []
    for name, value in point.items():
        lower, upper = bootstrap[name].quantile([0.025, 0.975]).to_numpy() if name not in counts else (np.nan, np.nan)
        rows.append({"metric": name, "value": value, "ci_lower": lower, "ci_upper": upper})
    return pd.DataFrame(rows)
