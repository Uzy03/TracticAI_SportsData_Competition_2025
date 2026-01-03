"""Script to build retrieval index using my_method backbone embeddings."""

import argparse
from pathlib import Path
from typing import Dict, Any
import yaml

from torch.utils.data import ConcatDataset

from my_method.retrieval import SimilarCKSearch
from my_method.dataio import ReceiverDataset, create_dataloader
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
    parser = argparse.ArgumentParser(description="Build retrieval index (my_method)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (used for model metadata)")
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        help="(Optional) Path to pretrained checkpoint (backbone-only or full). If omitted, auto-selects from config.",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        default=None,
        help="Path to save index file. If omitted, uses runs/my_method/<run_name>/indices/index_{d2|no_d2}.pkl",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=["train", "val", "test", None],
        help="Data phase to index. If omitted, uses all phases (train+val+test).",
    )
    parser.add_argument("--use-faiss", action="store_true", help="(Reserved) use faiss (index class supports it).")

    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))

    logger = setup_logging(
        config.get("log_dir", "runs/my_method"),
        config.get("log_level", "INFO"),
    )

    backbone_checkpoint_path = args.backbone_checkpoint or _auto_select_checkpoint(config)
    logger.info(f"Building retrieval index (my_method) on {device}")
    logger.info(f"Backbone checkpoint: {backbone_checkpoint_path}")
    d2_enabled = config.get("d2", {}).get("enabled", False)
    run_name = config.get("run_name", "default_run")
    default_out = Path("runs") / "my_method" / str(run_name) / "indices" / f"index_{'d2' if d2_enabled else 'no_d2'}.pkl"
    out_path = Path(args.output_index) if args.output_index else default_out
    logger.info(f"Output index: {out_path}")

    search_system = SimilarCKSearch(
        backbone_checkpoint_path=backbone_checkpoint_path,
        config=config,
        device=device,
    )

    # datasets
    datasets = []
    if args.phase is None:
        phases = ["train", "val", "test"]
        logger.info("Loading all phases: train, val, test")
    else:
        phases = [args.phase]
        logger.info(f"Loading phase: {args.phase}")

    phase_to_key = {"train": "train_path", "val": "val_path", "test": "test_path"}
    phase_to_key_multitask = {"train": "receiver_train_path", "val": "receiver_val_path", "test": "receiver_test_path"}

    for phase in phases:
        data_key = phase_to_key[phase]
        if "data" not in config or data_key not in config["data"]:
            data_key = phase_to_key_multitask[phase]
        data_path = config["data"][data_key]
        logger.info(f"  - {phase}: {data_path}")
        datasets.append(ReceiverDataset(data_path=data_path, file_format=config["data"].get("format", "pickle"), phase=phase))

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    logger.info(f"Total dataset loaded: {len(dataset)} samples")

    batch_size = config.get("eval", {}).get("batch_size", config.get("train", {}).get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    logger.info(f"Batch size: {batch_size}, num_workers: {num_workers}")

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    index = search_system.build_index(dataloader, index=None, save_path=out_path)
    logger.info(f"Index built successfully: {len(index)} embeddings")
    logger.info(f"Index saved to {out_path}")


if __name__ == "__main__":
    main()


