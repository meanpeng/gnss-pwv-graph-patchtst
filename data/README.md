# Data Directory

Raw and processed data are generated locally and are not committed.

Use:

```bash
python run.py --stage download --config configs/graph_patchtst.yaml
python run.py --stage preprocess --config configs/graph_patchtst.yaml
```

Expected local directories:

- `data/raw`: downloaded Zenodo files and extracted station text files
- `data/graph_processed`: aligned graph tensors and metadata
