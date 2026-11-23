"""Use ``python -m bsrt_neural_compression.run --help`` from Codes."""
from __future__ import annotations

import argparse
import json
import time

from .common import load_config, resolve


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a question-independent neural compressor")
    parser.add_argument("command", choices=["prepare", "train", "pilot", "complete", "validate", "final", "untouched", "plot", "status", "ablations"])
    parser.add_argument("--config", default="bsrt_neural_compression/config_pilot.json")
    args = parser.parse_args()
    config = load_config(resolve(args.config))
    if args.command == "prepare":
        from .data import split_manifest
        data = split_manifest(config)
        print(json.dumps({name: {"contexts": len(data[name]), "questions": sum(len(row["questions"]) for row in data[name])} for name in ["train", "validation", "untouched"]}, indent=2))
    elif args.command in {"train", "pilot", "complete"}:
        from .training import train
        from .common import seed_all
        seed_all(config["seed"], config["threads"])
        from .reward import QAReward
        qa = QAReward(config)
        train(config, qa=qa)
        if args.command == "pilot":
            from .evaluate import evaluate_development
            print(evaluate_development(config, qa=qa).to_string(index=False))
        qa.close()
        if args.command == "complete":
            del qa
            import gc
            gc.collect()
            from .evaluate import final_evaluation, evaluate_development
            final_evaluation(config)
            evaluate_development(config, "untouched")
        from .plot import plot_results
        plot_results(config)
    elif args.command in {"validate", "untouched", "final"}:
        from .common import seed_all
        seed_all(config["seed"], config["threads"])
        from .evaluate import evaluate_development, final_evaluation
        if args.command == "final":
            final_evaluation(config)
        else:
            evaluate_development(config, "validation" if args.command == "validate" else "untouched")
        from .plot import plot_results
        plot_results(config)
    elif args.command == "plot":
        from .plot import plot_results
        plot_results(config)
    elif args.command == "ablations":
        from .ablation import run_ablations
        run_ablations(config)
    else:
        output = resolve(config["output_dir"])
        for name in ["progress.json", "training_summary.json"]:
            path = output / name
            print(path.read_text(encoding="utf-8") if path.exists() else f"{name}: not yet available")


if __name__ == "__main__":
    main()
