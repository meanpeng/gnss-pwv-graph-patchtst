"""
Minimal CLI for the aligned graph GNSS-PWV forecasting release.

Supported stages:
download -> preprocess -> train -> evaluate
"""
import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.download import download_all
from src.data.graph_dataset import GraphPWVDataset, load_graph_processed_data
from src.data.graph_preprocess import preprocess_graph_all
from src.evaluate import evaluate
from src.train import train
from src.utils.config import load_config, override_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_experiment_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.experiment.name}_{timestamp}"
    exp_dir = os.path.join(config.experiment.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def build_graph_loader(config, split: str, shuffle: bool):
    data = load_graph_processed_data(config.data.processed_dir, split)
    stride = getattr(config.data, "stride", 1)
    if split in ("val", "test"):
        stride = getattr(config.data, "val_stride", stride)

    dataset = GraphPWVDataset(
        data,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        stride=stride,
        n_samples_per_epoch=(
            getattr(config.data, "train_samples_per_epoch", None)
            if split == "train"
            else None
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.experiment.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[Data] {split}: {len(dataset)} samples")
    return loader


def stage_download(config):
    download_all(config.data.raw_dir)


def stage_preprocess(config):
    preprocess_graph_all(
        raw_dir=config.data.raw_dir,
        processed_dir=config.data.processed_dir,
        stations_file=getattr(config.data, "stations_file", None),
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        missing_threshold=config.data.missing_threshold,
        impute_method=config.data.impute_method,
        normalize=config.data.normalize,
        scaler_type=config.data.scaler_type,
        min_common_hours=getattr(config.data, "graph_min_common_hours", 43800),
        min_num_nodes=getattr(config.data, "graph_min_num_nodes", 16),
        top_k=getattr(config.data, "graph_top_k", 6),
        sigma_km=getattr(config.data, "graph_distance_sigma_km", 350.0),
        graph_start_time=getattr(config.data, "graph_start_time", None),
        graph_end_time=getattr(config.data, "graph_end_time", None),
    )


def stage_train(config):
    train_loader = build_graph_loader(config, "train", shuffle=True)
    val_loader = build_graph_loader(config, "val", shuffle=False)
    exp_dir = make_experiment_dir(config)
    print(f"[Train] Experiment dir: {exp_dir}")
    train(config, train_loader, val_loader, exp_dir)
    return exp_dir


def stage_evaluate(config, checkpoint_path=None):
    if checkpoint_path is None:
        exp_root = config.experiment.output_dir
        exp_dirs = sorted(
            d for d in os.listdir(exp_root)
            if os.path.isdir(os.path.join(exp_root, d))
        )
        if not exp_dirs:
            raise FileNotFoundError(f"No experiment directories found in {exp_root}.")
        exp_dir = os.path.join(exp_root, exp_dirs[-1])
        checkpoint_path = os.path.join(exp_dir, "best_model.pt")
    else:
        exp_dir = os.path.dirname(checkpoint_path)

    print(f"[Eval] Checkpoint: {checkpoint_path}")
    test_loader = build_graph_loader(config, "test", shuffle=False)
    return evaluate(config, test_loader, checkpoint_path, exp_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Graph-PatchTST GNSS-PWV release")
    parser.add_argument(
        "--config",
        default="configs/graph_patchtst.yaml",
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--stage",
        default="train",
        choices=["download", "preprocess", "train", "evaluate", "all"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args, unknown = parser.parse_known_args()

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--") and i + 1 < len(unknown):
            key = unknown[i][2:]
            val = unknown[i + 1]
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                try:
                    val = float(val) if "." in val else int(val)
                except ValueError:
                    pass
            overrides[key] = val
            i += 2
        else:
            i += 1
    return args, overrides


def main():
    args, overrides = parse_args()
    config = load_config(args.config)
    if overrides:
        config = override_config(config, overrides)

    seed = args.seed if args.seed is not None else config.experiment.seed
    set_seed(seed)
    print(f"[Config] {args.config}")
    print(f"[Seed] {seed}")

    if args.stage in ("download", "all"):
        stage_download(config)
    if args.stage in ("preprocess", "all"):
        stage_preprocess(config)
    if args.stage in ("train", "all"):
        stage_train(config)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
