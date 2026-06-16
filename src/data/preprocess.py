"""
Preprocess PWV data: load, clean, impute, normalize, split, save.
Wu et al. (2025) China Coastal GNSS PWV dataset format:
    year month day hour min sec zhd_or_dummy ztd pwv
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Tuple


def load_station_data(raw_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all station PWV files from raw directory.
    Wu dataset: per-station .txt files with 9 whitespace-separated columns.
    Columns: year, month, day, hour, minute, second, col7, ztd, pwv
    """
    data_dir = os.path.join(raw_dir, "Final_pwv")
    if not os.path.exists(data_dir):
        data_dir = raw_dir
    
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    if not files:
        raise FileNotFoundError(
            f"No PWV data files found in {data_dir}. "
            "Please run download stage first."
        )
    
    stations = {}
    for f in files:
        # Skip metadata files
        basename = os.path.basename(f).lower()
        if "site" in basename or "readme" in basename or "documentation" in basename:
            continue
        
        station_id = os.path.splitext(os.path.basename(f))[0]
        
        # Read with whitespace delimiter, no header
        try:
            df = pd.read_csv(f, sep=r"\s+", header=None, engine="python")
        except Exception as e:
            print(f"[Preprocess] Warning: failed to read {station_id}: {e}")
            continue
        
        if df.shape[1] < 9:
            print(f"[Preprocess] Warning: {station_id} has only {df.shape[1]} columns, expected 9")
            continue
        
        # Build datetime from first 6 columns: year, month, day, hour, minute, second
        try:
            df["datetime"] = pd.to_datetime(
                df.iloc[:, :6].rename(columns={
                    0: "year", 1: "month", 2: "day",
                    3: "hour", 4: "minute", 5: "second"
                })
            )
        except Exception as e:
            print(f"[Preprocess] Warning: failed to parse datetime for {station_id}: {e}")
            continue
        
        # Column 8 (0-based index 8) is PWV
        df["pwv"] = pd.to_numeric(df.iloc[:, 8], errors="coerce")
        
        df = df[["datetime", "pwv"]].copy()
        df = df.dropna().sort_values("datetime").reset_index(drop=True)
        
        if len(df) > 0:
            stations[station_id] = df
            print(f"[Preprocess] Loaded {station_id}: {len(df)} records, "
                  f"from {df['datetime'].min()} to {df['datetime'].max()}")
    
    print(f"[Preprocess] Total loaded: {len(stations)} stations.")
    return stations


def clean_and_impute(
    df: pd.DataFrame,
    missing_threshold: float = 0.3,
    impute_method: str = "linear",
) -> pd.DataFrame:
    """Resample to hourly, handle missing values."""
    df = df.set_index("datetime")
    
    # Resample to hourly (mean if duplicates)
    df = df.resample("h").mean()
    
    missing_ratio = df["pwv"].isna().sum() / len(df)
    if missing_ratio > missing_threshold:
        return None  # Drop this station
    
    if impute_method == "linear":
        df["pwv"] = df["pwv"].interpolate(method="linear", limit_direction="both")
    elif impute_method == "forward":
        df["pwv"] = df["pwv"].ffill().bfill()
    
    df = df.dropna()
    return df.reset_index()


def split_series(
    values: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split."""
    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return values[:train_end], values[train_end:val_end], values[val_end:]


def preprocess_all(
    raw_dir: str,
    processed_dir: str,
    seq_len: int = 168,
    pred_len: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    missing_threshold: float = 0.3,
    impute_method: str = "linear",
    normalize: bool = True,
    scaler_type: str = "standard",
) -> Dict[str, any]:
    """
    Full preprocessing pipeline.
    Returns metadata dict with scaler and station info.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    stations = load_station_data(raw_dir)
    
    all_train, all_val, all_test = [], [], []
    station_meta = {}
    scalers = {}
    
    for station_id, df in stations.items():
        df = clean_and_impute(df, missing_threshold, impute_method)
        if df is None:
            print(f"[Preprocess] Skipping {station_id}: too many missing values (> {missing_threshold*100:.0f}%)")
            continue
        if len(df) < seq_len + pred_len + 100:
            print(f"[Preprocess] Skipping {station_id}: too short ({len(df)} < {seq_len + pred_len + 100})")
            continue
        
        values = df["pwv"].values.astype(np.float32)
        
        # Split
        train_vals, val_vals, test_vals = split_series(values, train_ratio, val_ratio)
        
        # Normalize (fit on train only)
        if normalize:
            if scaler_type == "standard":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            train_vals = scaler.fit_transform(train_vals.reshape(-1, 1)).flatten()
            val_vals = scaler.transform(val_vals.reshape(-1, 1)).flatten()
            test_vals = scaler.transform(test_vals.reshape(-1, 1)).flatten()
            scalers[station_id] = scaler
        
        all_train.append(train_vals)
        all_val.append(val_vals)
        all_test.append(test_vals)
        
        station_meta[station_id] = {
            "train_len": len(train_vals),
            "val_len": len(val_vals),
            "test_len": len(test_vals),
        }
        print(f"[Preprocess] Kept {station_id}: train={len(train_vals)}, val={len(val_vals)}, test={len(test_vals)}")
    
    if not all_train:
        raise ValueError("No valid stations after preprocessing.")
    
    # Save
    np.save(os.path.join(processed_dir, "train.npy"), np.array(all_train, dtype=object))
    np.save(os.path.join(processed_dir, "val.npy"), np.array(all_val, dtype=object))
    np.save(os.path.join(processed_dir, "test.npy"), np.array(all_test, dtype=object))
    
    with open(os.path.join(processed_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    
    with open(os.path.join(processed_dir, "meta.pkl"), "wb") as f:
        pickle.dump(station_meta, f)
    
    print(f"\n[Preprocess] Saved to {processed_dir}")
    print(f"[Preprocess] Valid stations: {len(all_train)} / {len(stations)} loaded")
    
    return {
        "scalers": scalers,
        "station_meta": station_meta,
        "num_stations": len(all_train),
    }


if __name__ == "__main__":
    preprocess_all("data/raw", "data/processed")
