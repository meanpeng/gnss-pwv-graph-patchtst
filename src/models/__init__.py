"""Model registry for the minimal Graph-PatchTST release."""
import os
import pickle

from .graph_patchtst import GraphPatchTST


def _load_graph_num_nodes(config, mcfg):
    num_nodes = getattr(mcfg, "num_nodes", None)
    if num_nodes is not None:
        return num_nodes

    meta_path = os.path.join(config.data.processed_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "num_nodes was not set and graph meta.pkl was not found. "
            f"Expected: {meta_path}"
        )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["num_nodes"]


def build_model_from_config(config):
    mcfg = config.model
    model_name = getattr(mcfg, "name", "graph_patchtst").lower()
    if model_name != "graph_patchtst":
        raise ValueError("This minimal release only supports model.name=graph_patchtst.")

    adjacency_path = getattr(
        mcfg,
        "adjacency_path",
        os.path.join(config.data.processed_dir, "adjacency.npy"),
    )
    return GraphPatchTST(
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        patch_len=getattr(mcfg, "patch_len", 24),
        stride=getattr(mcfg, "stride", 6),
        d_model=getattr(mcfg, "d_model", 64),
        n_heads=getattr(mcfg, "n_heads", 2),
        e_layers=getattr(mcfg, "e_layers", 2),
        d_ff=getattr(mcfg, "d_ff", 64),
        dropout=getattr(mcfg, "dropout", 0.1),
        num_nodes=_load_graph_num_nodes(config, mcfg),
        adjacency_path=adjacency_path,
        graph_layers=getattr(mcfg, "graph_layers", 1),
        graph_dropout=getattr(mcfg, "graph_dropout", getattr(mcfg, "dropout", 0.1)),
        graph_alpha_init=getattr(mcfg, "graph_alpha_init", 0.1),
        use_node_embedding=getattr(mcfg, "use_node_embedding", True),
    )
