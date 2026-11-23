"""Reproducible ablation suite; run with --prepare-only to inspect protocols."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import ROOT, digest, load_config, read_json, resolve, write_json
from .model import RetentionNetwork

SUITE = resolve("bsrt_neural_compression/results/controlled_ablation")
BASE = "bsrt_neural_compression/config_span_256_40.json"
LEVELS = list(range(5, 51, 5))
IDS = ["A0", "A1", "A2", "A3", "A4", "B0", "B2", "D0", "D1"]
LABELS = {"A0": "Full BSRT (40%-trained reference)", "A1": "No retention Transformer",
          "A2": "No budget conditioning (40%-trained)", "A3": "No structural features",
          "A4": "QA-reward-only (fresh initialization)", "B0": "Full BSRT (multi-budget, fresh)",
          "B2": "No budget conditioning (multi-budget, fresh)",
          "D0": "32-context answer-span reference", "D1": "32-context reward-distilled"}


def make_configs():
    base = load_config(resolve(BASE))
    base.pop("config_fingerprint")
    manifest = read_json(resolve(base["output_dir"]) / "splits.json")
    configurations = {}
    for name in IDS:
        config = copy.deepcopy(base)
        config.update(output_dir=str(SUITE / name), evaluation_percentages=LEVELS,
                      comparison_methods=["random", "l2", "attention", "learned_token_policy", "existing_question_oracle", "bsrt"],
                      plot_zero_reference=True, match_reference_initialization=True)
        config.pop("initialize_from", None)
        if name == "A1": config["model"]["contextual"] = False
        if name in {"A2", "B2"}: config["model"]["budget_conditioning"] = False
        if name == "A3": config["model"]["structural_features"] = False
        if name == "A4":
            config.update(pretrain_epochs=0, distill_epochs=8, improvement_rounds=1,
                          answer_span_supervision=False, initial_checkpoint_eligible=False,
                          auxiliary_weight=0.0, max_train_questions=1000000)
        if name in {"B0", "B2"}: config["removal_percentages"] = LEVELS
        source_manifest = copy.deepcopy(manifest)
        if name in {"D0", "D1"}:
            pilot = load_config(resolve("bsrt_neural_compression/config_pilot.json"))
            pilot.pop("config_fingerprint")
            pilot.update({key: config[key] for key in ["output_dir", "evaluation_percentages", "comparison_methods", "plot_zero_reference"]})
            config = pilot
            source_manifest = read_json(resolve("bsrt_neural_compression/results/pilot_40/splits.json"))
        if "initialize_from" in config:
            raise ValueError("Ablations must not have a warm start")
        folder = resolve(config["output_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "config.json"
        if path.exists() and read_json(path) != config:
            raise ValueError(f"Existing configuration differs: {name}")
        write_json(path, config)
        config = load_config(path)
        source_manifest["config_fingerprint"] = config["config_fingerprint"]
        write_json(folder / "splits.json", source_manifest)
        configurations[name] = config
    return configurations


def import_reference(name, config):
    source = {"A0": "span_256_40/checkpoints/best.pt",
              "D0": "pilot_40/checkpoints/best.pt",
              "D1": "pilot_40/checkpoints/distill_0_last.pt"}[name]
    path = resolve("bsrt_neural_compression/results") / source
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if name == "D1" and (saved["stage"] != "distill_0" or saved["epoch"] != 4):
        raise ValueError("Expected the saved genuinely distilled checkpoint")
    # Provenance records the original configuration before adapting output paths.
    provenance = {"source": str(path), "sha256": digest(path.read_bytes()),
                  "source_config": saved["config"], "stage": saved["stage"],
                  "epoch": saved["epoch"], "validation_f1": saved["validation_f1"]}
    destination = resolve(config["output_dir"])
    write_json(destination / "source_provenance.json", provenance)
    checkpoint = destination / "checkpoints/best.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.exists():
        saved["config"] = config
        torch.save(saved, checkpoint)
    parent = read_json(path.parents[1] / "training_summary.json")
    parent.update(best_stage=saved["stage"], best_epoch=saved["epoch"], best_validation_f1=saved["validation_f1"])
    write_json(destination / "training_summary.json", parent)
    if name == "A0":
        old_checkpoint = torch.load(resolve("bsrt_neural_compression/results/thesis_all_levels/checkpoints/best.pt"), weights_only=False, map_location="cpu")
        if not all(torch.equal(value, old_checkpoint["state_dict"][key]) for key, value in saved["state_dict"].items()):
            raise ValueError("Cannot reuse final A0 predictions: checkpoint weights differ")
        target = destination / "evaluation/final_500/bsrt_sample_results.csv"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolve("bsrt_neural_compression/results/thesis_all_levels/evaluation/final_500/bsrt_sample_results.csv"), target)


def run_worker(*args):
    subprocess.run([sys.executable, "-u", "-m", "bsrt_neural_compression.run", *args], cwd=ROOT, check=True)


def audit(configurations):
    rows = []
    primary = read_json(resolve(configurations["A0"]["output_dir"]) / "splits.json")
    for name, config in configurations.items():
        split = read_json(resolve(config["output_dir"]) / "splits.json")
        if name not in {"D0", "D1"}:
            for key in ["train", "validation"]:
                if split[key] != primary[key]:
                    raise ValueError(f"Mismatched {key} questions/contexts: {name}")
        network = RetentionNetwork(**config["model"])
        rows.append({"id": name, "label": LABELS[name], "train_contexts": len(split["train"]),
                     "train_questions": sum(len(row["questions"]) for row in split["train"]),
                     "training_budgets": config["removal_percentages"], "evaluation_budgets": LEVELS,
                     "epochs": config["distill_epochs"] if name == "A4" else config["pretrain_epochs"],
                     "seed": config["seed"], "learning_rate": config["learning_rate"],
                     "allocated_parameters": sum(p.numel() for p in network.parameters()),
                     "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
                     "warm_start": name == "D1", "existing_checkpoint_reused": name in {"A0", "D0", "D1"}})
    write_json(SUITE / "protocol.json", rows)


def collect(configurations):
    rows, paired = [], []
    reference = None
    a0 = resolve(configurations["A0"]["output_dir"]) / "evaluation/final_500/bsrt_sample_results.csv"
    if a0.exists(): reference = pd.read_csv(a0)
    rng = np.random.default_rng(20261103)
    for name, config in configurations.items():
        file = resolve(config["output_dir"]) / "evaluation/final_500/bsrt_sample_results.csv"
        if not file.exists(): continue
        frame = pd.read_csv(file)
        for level, part in frame.groupby("removal_percent"):
            if len(part) != 500: continue
            rows.append({"id": name, "removal_percent": level, "questions": len(part),
                         **{metric: part[metric].mean() for metric in ["accuracy", "f1", "semantic_similarity", "actual_token_reduction"]}})
        if reference is None or name not in {"A1", "A2", "A3", "A4"}: continue
        joined = reference[reference.removal_percent.eq(40)].merge(frame[frame.removal_percent.eq(40)], on=["question_id", "context_id"], suffixes=("_a0", "_variant"), validate="one_to_one")
        if len(joined) != 500: continue
        joined["difference"] = joined.f1_a0-joined.f1_variant
        clusters = joined.groupby("context_id").difference.agg(["sum", "count"])
        sums, counts = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
        selected = rng.integers(0, len(clusters), size=(10000, len(clusters)))
        low, high = np.quantile(sums[selected].sum(axis=1)/counts[selected].sum(axis=1), [.025, .975])
        signs = rng.choice([-1, 1], size=(50000, len(clusters)))
        observed = joined.difference.mean()
        p = (1+np.sum(np.abs(signs@sums/counts.sum()) >= abs(observed)-1e-12))/50001
        paired.append({"comparison": "A0 minus "+name, "removal_percent": 40,
                       "mean_f1_difference": observed, "ci_low": low, "ci_high": high, "cluster_signflip_p": p})
    summary = pd.DataFrame(rows)
    summary.to_csv(SUITE / "metrics.csv", index=False)
    tests = pd.DataFrame(paired)
    if len(tests):
        tests["holm_p"] = np.nan
        order = tests.cluster_signflip_p.sort_values().index
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (4-rank)*tests.loc[idx, "cluster_signflip_p"]))
            tests.loc[idx, "holm_p"] = running
        tests.to_csv(SUITE / "paired_f1_tests_40.csv", index=False)
    from .plot import markdown_table
    lines = ["# Controlled BSRT Ablation Study", "", "## Protocol", "",
             "Primary A0--A4: 256 identical training contexts, identical validation contexts/questions, 40% training budget, eight epochs/AdamW updates, learning rate 0.0005, weight decay 0.01, gradient clipping 1, and seed 20261103. All variants are evaluated on the same historical 500 QA pairs at 5--50%; levels other than 40% are budget-transfer results for this panel. A0 reuses the original successful, fresh-initialization answer-span checkpoint. New A1--A4 retention networks are fresh, with compatible shared modules initialized to identical values. The frozen QA encoder, features, tokenizer, cache, and whole-WordPiece selection are unchanged.", "",
             "A4 uses no answer-span labels and no warm start: training-only multi-question QA F1 mask search followed by eight reward-distillation epochs. It uses all training questions and zero answer-span auxiliary weight. The initial random head is not eligible to win checkpoint selection. Search adds computation and uses a different training objective; this is not an equal-compute comparison.", "",
             "B0 versus B2 separately tests budget conditioning: both fresh initializations, identical 5--50% training budgets, eight epochs, identical seed/data/optimizer and validation criterion. A2 at 40% alone measures removal of a constant budget input, not cross-budget adaptation.", "",
             "D0/D1 reuse the matched 32-context answer-span and best available saved reward-distilled checkpoints. They form a separate historical stage comparison and must not be pooled with the 256-context causal ablations. D1 received additional search/training, so its compute is not matched to D0. No new reward-distilled model is retrained for this comparison.", "",
             "The final 500 have been examined in previous thesis work. Checkpoints here are selected only on validation; final results are not used to choose epochs. Single training seed limits claims about variability between training runs. F1 is the preregistered primary endpoint at 40%. Intervals resample contexts (10,000 bootstrap draws); two-sided context sign-flip tests use 50,000 draws and Holm correction over the four primary A0 contrasts, alpha=0.05. Sign-flip inference assumes cluster-level symmetry under the null.", "",
             "## Measured results", "", markdown_table(summary) if len(summary) else "Experiments have not completed yet.", ""]
    if len(tests): lines += ["## Primary paired F1 comparisons at 40%", "", markdown_table(tests), ""]
    lines += ["## Status", "", "Only completed 500-question levels enter the tables. Missing variants are pending, not zero scores. Parameter counts and exact run configurations are in protocol.json and each variant/config.json. Checkpoints and original source provenance remain separate."]
    (SUITE / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if len(summary): plot_summary(summary)


def plot_summary(frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    archive = resolve("../results_attention_balanced_token_policy/standard_methods_sample_results.csv")
    reference = pd.read_csv(archive, usecols=["squad_id", "original_accuracy", "original_f1", "original_semantic_similarity"]).drop_duplicates("squad_id")
    groups = [("primary", ["A0", "A1", "A2", "A3", "A4"]), ("budget", ["B0", "B2"]), ("distillation", ["D0", "D1"])]
    for panel, ids in groups:
        if not frame.id.isin(ids).any():
            continue
        for metric in ["accuracy", "f1", "semantic_similarity"]:
            fig, ax = plt.subplots(figsize=(9, 6))
            for name in ids:
                part = frame[frame.id.eq(name)].sort_values("removal_percent")
                if len(part): ax.plot([0]+part.removal_percent.tolist(), [reference["original_"+metric].mean()]+part[metric].tolist(), marker="o", label=name+": "+LABELS[name], linewidth=2, markersize=4)
            ax.set(xlabel="Target token removal (%)", ylabel={"f1": "Answer F1", "accuracy": "Accuracy", "semantic_similarity": "Semantic similarity"}[metric], ylim=(0, 1), xticks=[0, 10, 20, 30, 40, 50])
            ax.grid(alpha=.2)
            ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.28), frameon=False, fontsize=8.5, ncol=2 if len(ids)>2 else 1)
            fig.tight_layout()
            for suffix in ["pdf", "svg", "png"]: fig.savefig(SUITE / f"{panel}_{metric}.{suffix}", dpi=180, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    configurations = make_configs()
    audit(configurations)
    if args.report_only:
        collect(configurations)
        return
    for name in ["A0", "D0", "D1"]: import_reference(name, configurations[name])
    collect(configurations)
    if args.prepare_only:
        print("Prepared controlled study:", SUITE, flush=True)
        return
    for name in IDS:
        config = configurations[name]
        path = str(resolve(config["output_dir"]) / "config.json")
        print(f"ABLATION {name}: {LABELS[name]}", flush=True)
        gc.collect()
        if name not in {"A0", "D0", "D1"}:
            run_worker("train", "--config", path)
        run_worker("final", "--config", path)
        collect(configurations)
    print("Completed controlled BSRT ablations:", SUITE, flush=True)


if __name__ == "__main__":
    main()
