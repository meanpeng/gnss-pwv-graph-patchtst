"""
Graph-mode preprocessing for aligned multi-station PWV forecasting.

This module is intentionally separate from the original per-station
preprocessing pipeline. It writes a different processed directory containing
network-aligned tensors and a static geographic adjacency matrix.
"""
import json
import os
import pickle
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .preprocess import clean_and_impute, load_station_data


def parse_station_metadata(stations_file: str) -> Dict[str, Dict[str, float]]:
    """Parse CGN_sites.txt into a compact station metadata dictionary."""
    metadata = {}
    if not stations_file or not os.path.exists(stations_file):
        return metadata

    with open(stations_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        station_id = parts[0]
        try:
            metadata[station_id] = {
                "lat": float(parts[1]),
                "lon": float(parts[2]),
                "height_m": float(parts[3]),
                "metadata_mean_mm": float(parts[5]),
                "metadata_rms_mm": float(parts[6]),
            }
        except ValueError:
            continue
    return metadata


def _span_hours(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if start is None or end is None or end < start:
        return 0
    return int((end - start) / pd.Timedelta(hours=1)) + 1


def _common_span(station_ids: Iterable[str], cleaned: Dict[str, pd.DataFrame]):
    starts = [cleaned[sid]["datetime"].min() for sid in station_ids]
    ends = [cleaned[sid]["datetime"].max() for sid in station_ids]
    common_start = max(starts)
    common_end = min(ends)
    return common_start, common_end, _span_hours(common_start, common_end)


def select_common_stations(
    cleaned: Dict[str, pd.DataFrame],
    min_common_hours: int,
    min_num_nodes: int,
    graph_start_time: str = None,
    graph_end_time: str = None,
) -> Tuple[List[str], pd.Timestamp, pd.Timestamp]:
    """
    Select stations with a sufficiently long common overlapping time span.

    If explicit start/end times are provided, only stations covering that span
    are retained. Otherwise, stations that constrain the common overlap are
    greedily removed until the requested overlap is reached or min_num_nodes is
    hit.
    """
    selected = list(cleaned.keys())
    if not selected:
        raise ValueError("No cleaned stations available for graph preprocessing.")

    if graph_start_time or graph_end_time:
        if not graph_start_time or not graph_end_time:
            raise ValueError("Both graph_start_time and graph_end_time must be set together.")
        common_start = pd.Timestamp(graph_start_time)
        common_end = pd.Timestamp(graph_end_time)
        selected = [
            sid
            for sid in selected
            if cleaned[sid]["datetime"].min() <= common_start
            and cleaned[sid]["datetime"].max() >= common_end
        ]
        if len(selected) < min_num_nodes:
            raise ValueError(
                f"Only {len(selected)} stations cover {common_start} to {common_end}; "
                f"min_num_nodes={min_num_nodes}."
            )
        return selected, common_start, common_end

    while len(selected) > min_num_nodes:
        common_start, common_end, hours = _common_span(selected, cleaned)
        if hours >= min_common_hours:
            return selected, common_start, common_end

        best_sid = None
        best_hours = hours
        best_span = (common_start, common_end)
        for sid in selected:
            trial = [item for item in selected if item != sid]
            trial_start, trial_end, trial_hours = _common_span(trial, cleaned)
            if trial_hours > best_hours:
                best_sid = sid
                best_hours = trial_hours
                best_span = (trial_start, trial_end)

        if best_sid is None:
            break
        selected.remove(best_sid)

    common_start, common_end, hours = _common_span(selected, cleaned)
    if hours <= 0:
        raise ValueError("Selected graph stations have no common overlapping span.")
    return selected, common_start, common_end


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers."""
    radius_km = 6371.0088
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def build_distance_adjacency(
    station_ids: List[str],
    station_metadata: Dict[str, Dict[str, float]],
    top_k: int = 6,
    sigma_km: float = 350.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a symmetric top-k distance adjacency with self loops."""
    missing = [sid for sid in station_ids if sid not in station_metadata]
    if missing:
        raise ValueError(f"Missing station metadata for adjacency: {missing}")

    coords = np.array(
        [[station_metadata[sid]["lat"], station_metadata[sid]["lon"]] for sid in station_ids],
        dtype=np.float32,
    )
    n_nodes = len(station_ids)
    distances = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        distances[i] = haversine_km(coords[i, 0], coords[i, 1], coords[:, 0], coords[:, 1])

    sigma_km = max(float(sigma_km), 1e-6)
    weights = np.exp(-((distances / sigma_km) ** 2)).astype(np.float32)
    np.fill_diagonal(weights, 0.0)

    adjacency = np.zeros_like(weights)
    if top_k is None or top_k <= 0 or top_k >= n_nodes - 1:
        adjacency = weights
    else:
        for i in range(n_nodes):
            neighbors = np.argsort(distances[i])[1 : top_k + 1]
            adjacency[i, neighbors] = weights[i, neighbors]

    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency.astype(np.float32), distances.astype(np.float32)


def build_identity_adjacency(n_nodes: int) -> np.ndarray:
    """Identity graph: self-loop only, no spatial mixing."""
    adjacency = np.eye(n_nodes, dtype=np.float32)
    return adjacency


def build_random_adjacency(
    n_nodes: int,
    num_undirected_edges: int,
    seed: int = 42,
) -> np.ndarray:
    """Random graph with the same number of undirected edges as the reference."""
    rng = np.random.RandomState(seed)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    upper_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    chosen = rng.choice(len(upper_pairs), size=min(num_undirected_edges, len(upper_pairs)), replace=False)
    for idx in chosen:
        i, j = upper_pairs[idx]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def preprocess_graph_all(
    raw_dir: str,
    processed_dir: str,
    stations_file: str,
    seq_len: int = 168,
    pred_len: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    missing_threshold: float = 0.3,
    impute_method: str = "linear",
    normalize: bool = True,
    scaler_type: str = "standard",
    min_common_hours: int = 43800,
    min_num_nodes: int = 16,
    top_k: int = 6,
    sigma_km: float = 350.0,
    graph_start_time: str = None,
    graph_end_time: str = None,
) -> Dict[str, object]:
    """
    Build graph-mode train/val/test arrays with shape [time, station].

    The original station-wise processed files are not touched. All outputs are
    written under processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)

    raw_stations = load_station_data(raw_dir)
    station_metadata = parse_station_metadata(stations_file)

    cleaned = {}
    dropped = {}
    for station_id, df in raw_stations.items():
        graph_df = clean_and_impute(df, missing_threshold, impute_method)
        if graph_df is None:
            dropped[station_id] = "missing_threshold"
            continue
        if len(graph_df) < seq_len + pred_len + 100:
            dropped[station_id] = "too_short"
            continue
        if station_id not in station_metadata:
            dropped[station_id] = "missing_metadata"
            continue
        cleaned[station_id] = graph_df

    if not cleaned:
        raise ValueError("No valid stations after graph preprocessing filters.")

    selected, common_start, common_end = select_common_stations(
        cleaned=cleaned,
        min_common_hours=min_common_hours,
        min_num_nodes=min_num_nodes,
        graph_start_time=graph_start_time,
        graph_end_time=graph_end_time,
    )
    selected = sorted(selected)

    common_index = pd.date_range(common_start, common_end, freq="h")
    matrix_columns = []
    for station_id in selected:
        df = cleaned[station_id].set_index("datetime").sort_index()
        series = df["pwv"].reindex(common_index)
        if series.isna().any():
            series = series.interpolate(method="time", limit_direction="both").ffill().bfill()
        matrix_columns.append(series.to_numpy(dtype=np.float32))

    values = np.stack(matrix_columns, axis=1).astype(np.float32)
    if values.shape[0] < seq_len + pred_len + 100:
        raise ValueError(
            f"Aligned graph series is too short: {values.shape[0]} hours for "
            f"seq_len={seq_len}, pred_len={pred_len}."
        )

    n_total = values.shape[0]
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    train_vals = values[:train_end]
    val_vals = values[train_end:val_end]
    test_vals = values[val_end:]

    scalers = {}
    if normalize:
        train_norm = np.zeros_like(train_vals, dtype=np.float32)
        val_norm = np.zeros_like(val_vals, dtype=np.float32)
        test_norm = np.zeros_like(test_vals, dtype=np.float32)
        for i, station_id in enumerate(selected):
            scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
            train_norm[:, i] = scaler.fit_transform(train_vals[:, i : i + 1]).reshape(-1)
            val_norm[:, i] = scaler.transform(val_vals[:, i : i + 1]).reshape(-1)
            test_norm[:, i] = scaler.transform(test_vals[:, i : i + 1]).reshape(-1)
            scalers[station_id] = scaler
        train_vals, val_vals, test_vals = train_norm, val_norm, test_norm

    adjacency, distances = build_distance_adjacency(
        selected,
        station_metadata=station_metadata,
        top_k=top_k,
        sigma_km=sigma_km,
    )

    np.save(os.path.join(processed_dir, "train.npy"), train_vals.astype(np.float32))
    np.save(os.path.join(processed_dir, "val.npy"), val_vals.astype(np.float32))
    np.save(os.path.join(processed_dir, "test.npy"), test_vals.astype(np.float32))
    np.save(os.path.join(processed_dir, "adjacency.npy"), adjacency)
    np.save(os.path.join(processed_dir, "distances_km.npy"), distances)

    with open(os.path.join(processed_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    selected_meta = {
        sid: {
            **station_metadata[sid],
            "raw_start": str(cleaned[sid]["datetime"].min()),
            "raw_end": str(cleaned[sid]["datetime"].max()),
        }
        for sid in selected
    }
    meta = {
        "dataset_mode": "graph",
        "station_ids": selected,
        "num_nodes": len(selected),
        "common_start": str(common_start),
        "common_end": str(common_end),
        "common_hours": int(values.shape[0]),
        "train_len": int(train_vals.shape[0]),
        "val_len": int(val_vals.shape[0]),
        "test_len": int(test_vals.shape[0]),
        "station_meta": selected_meta,
        "dropped_stations": dropped,
        "adjacency": {
            "type": "distance_topk",
            "top_k": top_k,
            "sigma_km": sigma_km,
            "path": os.path.join(processed_dir, "adjacency.npy"),
        },
    }
    with open(os.path.join(processed_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    with open(os.path.join(processed_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[GraphPreprocess] Saved to {processed_dir}")
    print(
        f"[GraphPreprocess] Stations: {len(selected)} | "
        f"Common span: {common_start} to {common_end} ({values.shape[0]} hours)"
    )
    print(
        f"[GraphPreprocess] Split: train={train_vals.shape}, "
        f"val={val_vals.shape}, test={test_vals.shape}"
    )
    print(f"[GraphPreprocess] Adjacency: top_k={top_k}, sigma_km={sigma_km}")

    return meta
