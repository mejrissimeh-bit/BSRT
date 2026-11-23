"""Evaluate the best genuinely multi-budget-trained checkpoint separately."""
from __future__ import annotations

import argparse
import copy

import torch

from .common import digest, load_config, read_json, resolve, seed_all, write_json

SOURCE_CONFIG = "bsrt_neural_compression/config_thesis_all_levels.json"
OUTPUT = "bsrt_neural_compression/results/multibudget_epoch1_comparison"


def prepare():
    source_config = load_config(resolve(SOURCE_CONFIG))
    source_dir = resolve(source_config["output_dir"])
    source_path = source_dir / "checkpoints" / "pretrain_best.pt"
    saved = torch.load(source_path, map_location="cpu", weights_only=False)
    if saved["stage"] != "pretrain" or saved["epoch"] < 1:
        raise ValueError("Expected an actually multi-budget-trained checkpoint")
    config = copy.deepcopy(source_config)
    config.pop("config_fingerprint")
    config["output_dir"] = OUTPUT
    config["comparison_methods"] = ["random", "attention", "l2", "learned_token_policy", "existing_question_oracle", "bsrt"]
    config["plot_zero_reference"] = True
    config["plot_formats"] = ["svg", "png", "pdf"]
    destination = resolve(OUTPUT)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", config)
    config = load_config(destination / "config.json")
    provenance = {"source_checkpoint": str(source_path), "source_sha256": digest(source_path.read_bytes()),
                  "trained_stage": saved["stage"], "epoch": saved["epoch"],
                  "validation_mean_f1": saved["validation_f1"],
                  "selection": "Best validation score among multi-budget-trained epochs; excludes the initial 40%-trained checkpoint."}
    metadata_path = destination / "checkpoint_source.json"
    if metadata_path.exists() and read_json(metadata_path) != provenance:
        raise ValueError("The separate comparison is already frozen to another checkpoint")
    write_json(metadata_path, provenance)
    manifest = read_json(source_dir / "splits.json")
    manifest["config_fingerprint"] = config["config_fingerprint"]
    write_json(destination / "splits.json", manifest)
    checkpoint_path = destination / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(exist_ok=True)
        saved["config"] = config
        torch.save(saved, checkpoint_path)
    summary = read_json(source_dir / "training_summary.json")
    summary.update(best_stage=saved["stage"], best_epoch=saved["epoch"], best_validation_f1=saved["validation_f1"])
    write_json(destination / "training_summary.json", summary)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    config = prepare()
    if args.prepare_only:
        print(resolve(OUTPUT), flush=True)
        return
    seed_all(config["seed"], config["threads"])
    if not args.plot_only:
        from .evaluate import final_evaluation
        final_evaluation(config)
    from .plot import plot_results
    plot_results(config)
    print(f"Completed multi-budget comparison: {resolve(OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
