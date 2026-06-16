"""Evaluation for aligned graph GNSS-PWV forecasts."""
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from src.models import build_model_from_config
from src.utils.metrics import compute_metrics


def get_device(config):
    device_cfg = config.training.get("device", "auto") if hasattr(config, "training") else "auto"
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_scalers(processed_dir):
    scaler_path = os.path.join(processed_dir, "scalers.pkl")
    if not os.path.exists(scaler_path):
        return None
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def load_graph_meta(processed_dir):
    meta_path = os.path.join(processed_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def inverse_transform_graph(preds, trues, scalers, station_ids=None):
    if scalers is None:
        return preds, trues

    if isinstance(scalers, dict):
        if station_ids is None:
            station_ids = list(scalers.keys())
        scaler_list = [scalers[sid] for sid in station_ids]
    else:
        scaler_list = scalers

    preds_inv = np.zeros_like(preds)
    trues_inv = np.zeros_like(trues)
    for node_idx, scaler in enumerate(scaler_list):
        p_flat = preds[:, :, node_idx].reshape(-1, 1)
        t_flat = trues[:, :, node_idx].reshape(-1, 1)
        preds_inv[:, :, node_idx] = scaler.inverse_transform(p_flat).reshape(
            preds[:, :, node_idx].shape
        )
        trues_inv[:, :, node_idx] = scaler.inverse_transform(t_flat).reshape(
            trues[:, :, node_idx].shape
        )
    return preds_inv, trues_inv


def evaluate(config, test_loader, checkpoint_path, exp_dir):
    device = get_device(config)
    model = build_model_from_config(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluate", leave=False):
            x = x.to(device, non_blocking=True)
            out = model(x)
            all_preds.append(out.cpu().numpy())
            all_trues.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)

    meta = load_graph_meta(config.data.processed_dir)
    station_ids = meta.get("station_ids") if meta else None
    preds_mm, trues_mm = inverse_transform_graph(
        preds,
        trues,
        load_scalers(config.data.processed_dir),
        station_ids=station_ids,
    )

    metrics = compute_metrics(
        preds_mm,
        trues_mm,
        getattr(config.experiment, "metrics", ["mse", "mae", "rmse", "mape"]),
    )
    os.makedirs(exp_dir, exist_ok=True)

    metrics_path = os.path.join(exp_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Metrics saved to {metrics_path}")
    for key, value in metrics.items():
        print(f"[Eval] {key}: {value:.6f}" if isinstance(value, float) else f"[Eval] {key}: {value}")

    if getattr(config.experiment, "save_predictions", True):
        pred_path = os.path.join(exp_dir, "predictions.npz")
        np.savez_compressed(pred_path, preds=preds_mm, trues=trues_mm)
        print(f"[Eval] Predictions saved to {pred_path}")

    return metrics
