"""
Evaluation metrics for time series forecasting.
"""
import numpy as np
import torch


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    mask = true != 0
    return float(np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100)


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def bias(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean prediction bias (positive = over-prediction)."""
    return float(np.mean(pred - true))


def compute_metrics(pred: np.ndarray, true: np.ndarray, metrics_list=None):
    """Compute a dict of metrics."""
    if metrics_list is None:
        metrics_list = ["mse", "mae", "rmse", "mape"]
    
    results = {}
    for metric in metrics_list:
        if metric == "mse":
            results["mse"] = mse(pred, true)
        elif metric == "mae":
            results["mae"] = mae(pred, true)
        elif metric == "rmse":
            results["rmse"] = rmse(pred, true)
        elif metric == "mape":
            results["mape"] = mape(pred, true)
        elif metric == "r2":
            results["r2"] = r2_score(pred, true)
        elif metric == "bias":
            results["bias"] = bias(pred, true)
    
    return results
