# Data Protocol

This release uses the aligned graph protocol, not the older independent
single-station window protocol.

## Raw Data

The downloader fetches these public Zenodo files into `data/raw`:

- `Final_pwv.zip`
- `CGN_sites.txt`

The raw data are not committed to the repository.

## Preprocessing

`src/data/graph_preprocess.py` performs the graph preprocessing path:

1. Load per-station hourly PWV files.
2. Clean and linearly impute station time series.
3. Retain stations with sufficient common temporal coverage and metadata.
4. Align all retained stations onto one common hourly time axis.
5. Split chronologically into train, validation, and test tensors.
6. Normalize each station using training data statistics.
7. Build a static geographic adjacency matrix from station coordinates.

The default processed output is `data/graph_processed`.

## Default Benchmark Shape

With `configs/graph_patchtst.yaml`, the intended benchmark has:

- input length: 168 hours
- prediction length: 24 hours
- train/validation/test split: chronological
- validation/test stride: 6
- train samples per epoch: 47470
- retained station count: 30 under the current manuscript settings

Generated files include:

- `train.npy`
- `val.npy`
- `test.npy`
- `adjacency.npy`
- `adjacency_k15_sigma800.npy`
- `distances_km.npy`
- `scalers.pkl`
- `meta.pkl`
- `meta.json`

## Evaluation

Evaluation loads `scalers.pkl` and reports metrics after inverse transforming
predictions and targets back to PWV millimeters.

This preprocessing is an offline benchmark construction. It should not be
described as a strict causal real-time imputation pipeline.
