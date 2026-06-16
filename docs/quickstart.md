# Quickstart

From the repository root:

```bash
pip install -r requirements.txt
python run.py --stage download --config configs/graph_patchtst.yaml
python run.py --stage preprocess --config configs/graph_patchtst.yaml
python run.py --stage train --config configs/graph_patchtst.yaml
python run.py --stage evaluate --config configs/graph_patchtst.yaml --checkpoint experiments/main_results/<run_dir>/best_model.pt
```

For a quick smoke test, reduce epochs and workers from the command line:

```bash
python run.py --stage train --config configs/graph_patchtst.yaml --training.epochs 1 --experiment.num_workers 0
```

The training command writes checkpoints and logs under `experiments/main_results`.
