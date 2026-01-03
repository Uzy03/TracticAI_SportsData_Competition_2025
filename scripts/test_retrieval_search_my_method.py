"""Test script for my_method similar CK retrieval (Top/Bottom-k)."""

import argparse
from typing import Dict, Any
import yaml
import numpy as np

from my_method.retrieval import SimilarCKSearch, SimilarCKIndex
from my_method.dataio import ReceiverDataset
from my_method.modules import setup_logging, get_device


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _auto_select_checkpoint(config: Dict[str, Any]) -> str:
    d2_enabled = config.get("d2", {}).get("enabled", False)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    model_save_dir = config.get("model_save_dir", f"{checkpoint_dir}/receiver_shot")
    run_name = config.get("run_name", None)
    if run_name:
        model_save_dir = f"{model_save_dir}/{run_name}"
    return f"{model_save_dir}/best_d2.ckpt" if d2_enabled else f"{model_save_dir}/best_no_d2.ckpt"


def main():
    parser = argparse.ArgumentParser(description="Test retrieval search (my_method)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (used for model metadata)")
    parser.add_argument("--index-path", type=str, required=True, help="Path to index file")
    parser.add_argument("--query-index", type=int, default=0, help="Query sample index (default: 0)")
    parser.add_argument("--k", type=int, default=5, help="Top/Bottom-k (default: 5)")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "val", "test"], help="Query phase")
    parser.add_argument("--data-path", type=str, help="Override query data path")
    parser.add_argument("--backbone-checkpoint", type=str, help="Optional checkpoint path (otherwise auto-select)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    logger = setup_logging(config.get("log_dir", "runs/my_method"), config.get("log_level", "INFO"))

    backbone_checkpoint_path = args.backbone_checkpoint or _auto_select_checkpoint(config)
    logger.info(f"Testing retrieval search (my_method) on {device}")
    logger.info(f"Backbone checkpoint: {backbone_checkpoint_path}")
    logger.info(f"Index path: {args.index_path}")

    search_system = SimilarCKSearch(backbone_checkpoint_path=backbone_checkpoint_path, config=config, device=device)

    index = SimilarCKIndex(embedding_dim=config["model"]["hidden_dim"], index_path=args.index_path)
    index.load(args.index_path)
    logger.info(f"Index loaded: {len(index)} embeddings")

    if args.data_path:
        data_path = args.data_path
    else:
        phase_to_key = {"train": "train_path", "val": "val_path", "test": "test_path"}
        phase_to_key_multitask = {"train": "receiver_train_path", "val": "receiver_val_path", "test": "receiver_test_path"}
        key = phase_to_key[args.phase]
        if "data" not in config or key not in config["data"]:
            key = phase_to_key_multitask[args.phase]
        data_path = config["data"][key]

    ds = ReceiverDataset(data_path=data_path, file_format=config["data"].get("format", "pickle"), phase=args.phase)
    if args.query_index >= len(ds):
        raise ValueError(f"Query index {args.query_index} out of range (dataset size={len(ds)})")

    query_data, query_target = ds[args.query_index]
    logger.info(f"Query receiver target: {int(query_target.item())}")

    # Compute similarities against all index embeddings (index embeddings are normalized)
    x = query_data["x"].to(search_system.device)
    edge_index = query_data["edge_index"].to(search_system.device)
    edge_attr = query_data.get("edge_attr")
    if edge_attr is not None:
        edge_attr = edge_attr.to(search_system.device)
    batch = query_data.get("batch")
    if batch is not None:
        batch = batch.to(search_system.device)

    with np.errstate(all="ignore"):
        import torch
        with torch.no_grad():
            q = search_system._forward_batch(x, edge_index, edge_attr, batch).detach().cpu().numpy().reshape(1, -1)
        q = q.astype(np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        sims = np.dot(q, index.embeddings.T).reshape(-1)

    n = int(len(sims))
    k = int(min(int(args.k), n))
    top_idx = np.argsort(sims)[::-1][:k]
    bot_idx = np.argsort(sims)[:k]

    print("\n" + "=" * 80)
    print(f"Query index: {args.query_index} | phase={args.phase} | data={data_path}")
    print(f"Target receiver: {int(query_target.item())}")
    print("=" * 80)

    print(f"\nTop-{k}:")
    for rank, i in enumerate(top_idx, 1):
        print(f"{rank:>2}. idx={int(i):>4} sim={float(sims[i]):.8f}")

    print(f"\nBottom-{k}:")
    for rank, i in enumerate(bot_idx, 1):
        print(f"{rank:>2}. idx={int(i):>4} sim={float(sims[i]):.8f}")


if __name__ == "__main__":
    main()


