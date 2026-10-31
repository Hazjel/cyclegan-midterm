"""Evaluate a selected CycleGAN checkpoint on testA/testB once, without training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from cyclegan import ArchitectureConfig, Generator, PatchDiscriminator, create_dataloaders
from train_cyclegan import choose_device, test_and_samples


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--dataset-root", type=Path, default=REPO_ROOT.parent / "data" / "horse2zebra")
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs" / "evaluations")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("batch-size must be positive and num-workers nonnegative")
    checkpoint_path = args.checkpoint.resolve()
    root = args.dataset_root.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {root / 'manifest.json'}")

    device = choose_device(args.device)
    # Checkpoints also contain optimizer state, which evaluation does not use.
    # Keep the checkpoint payload on CPU and move only the four models to GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ArchitectureConfig(**checkpoint["architecture"])
    models = [Generator(config), Generator(config), PatchDiscriminator(config), PatchDiscriminator(config)]
    for model, key in zip(models, ("G_AB", "G_BA", "D_A", "D_B")):
        model.load_state_dict(checkpoint["model_state_dicts"][key], strict=True)
        model.to(device).eval()

    loaders = create_dataloaders(root, batch_size=args.batch_size,
                                 num_workers=args.num_workers, seed=checkpoint["seed"])
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    output = args.output_root.resolve() / f"evaluation-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    output.mkdir(exist_ok=False)
    metrics = test_and_samples(loaders, models, device, output)
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "evaluation_config.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"], "dataset_root": str(root),
        "device": str(device), "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "manifest_sha256": checkpoint.get("metadata", {}).get("manifest_sha256")}, indent=2), encoding="utf-8")
    print(f"Test evaluation saved to: {output}")


if __name__ == "__main__":
    main()
