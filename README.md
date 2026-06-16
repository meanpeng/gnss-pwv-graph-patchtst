# Graph-PatchTST for GNSS-PWV Forecasting

This is a minimal release for aligned graph-based GNSS precipitable water vapor
(PWV) forecasting over China's coastal station network.

The release includes only the reproducible core:

- download the public Wu et al. (2025) coastal GNSS-PWV dataset from Zenodo
- preprocess the data into a 30-station aligned graph benchmark
- train Graph-PatchTST
- evaluate a checkpoint on the test split with inverse-normalized PWV metrics

It intentionally excludes manuscript-production scripts, ablation code,
statistical diagnostics, figure rendering scripts, checkpoints, and local
experiment outputs.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the raw dataset:

```bash
python run.py --stage download --config configs/graph_patchtst.yaml
```

Preprocess the aligned graph dataset:

```bash
python run.py --stage preprocess --config configs/graph_patchtst.yaml
```

Train Graph-PatchTST:

```bash
python run.py --stage train --config configs/graph_patchtst.yaml
```

Evaluate a checkpoint:

```bash
python run.py --stage evaluate --config configs/graph_patchtst.yaml --checkpoint experiments/main_results/<run_dir>/best_model.pt
```

Run the full pipeline after the raw data is available:

```bash
python run.py --stage all --config configs/graph_patchtst.yaml
```

## Data Protocol

The default config builds the aligned graph benchmark:

- processed directory: `data/graph_processed`
- input window: 168 hourly PWV values
- forecast horizon: 24 hourly PWV values
- graph node count: 30 retained stations under the default filtering settings
- model input/output shape: `[batch, time, station]`
- evaluation unit: inverse-normalized PWV in mm

See `docs/data_protocol.md` for details.

## Scope

This minimal release is not a full benchmark suite. It does not include code for
baseline sweeps, graph ablations, bootstrap confidence intervals, high-PWV event
metrics, seasonal diagnostics, spatial diagnostics, or manuscript figures.

Those analyses were used for the paper, but the public code surface here is
limited to data preprocessing, model training, and validation/testing.

## Data Source

Wu et al. (2025), China coastal GNSS PWV dataset:

https://doi.org/10.5281/zenodo.17012498

## Citation

If you use this code, cite the original GNSS-PWV dataset and the associated
Graph-PatchTST manuscript when available.
