"""Three input-only ablations against the successful 40%-trained BSRT."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from .common import ROOT, load_config, read_json, resolve, write_json
from .model import RetentionNetwork
from .plot import markdown_table


SUITE = resolve("bsrt_neural_compression/results/three_input_ablations")
PREVIOUS = resolve("bsrt_neural_compression/results/controlled_ablation")
LABELS = {"full": "Full BSRT", "no_contextual_vector": "Without contextual BERT vector",
          "no_learned_position": "Without learned position embedding", "no_budget": "Without budget embedding"}


def prepare():
    base = read_json(PREVIOUS / "A0/config.json")
    source_split = read_json(PREVIOUS / "A0/splits.json")
    configs = {"full": load_config(PREVIOUS / "A0/config.json"),
               "no_budget": load_config(PREVIOUS / "A2/config.json")}
    SUITE.mkdir(parents=True, exist_ok=True)
    for name, flag in [("no_contextual_vector", "contextual_vectors"),
                       ("no_learned_position", "position_embedding")]:
        config = copy.deepcopy(base)
        config.update(output_dir=str(SUITE / name), match_reference_initialization=True)
        config.pop("initialize_from", None)
        config["model"][flag] = False
        folder = resolve(config["output_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "config.json"
        if path.exists() and read_json(path) != config:
            raise ValueError(f"Configuration changed on resume: {name}")
        write_json(path, config)
        configs[name] = load_config(path)
        split = copy.deepcopy(source_split)
        split["config_fingerprint"] = configs[name]["config_fingerprint"]
        write_json(folder / "splits.json", split)
    protocol = []
    for name in LABELS:
        config = configs[name]
        split = read_json(resolve(config["output_dir"]) / "splits.json")
        if any(split[key] != source_split[key] for key in ["train", "validation"]):
            raise ValueError(f"Different data for {name}")
        model = RetentionNetwork(**config["model"])
        protocol.append({"id": name, "label": LABELS[name], "config": config,
                         "reused_completed_run": name in {"full", "no_budget"},
                         "warm_start": False, "train_contexts": len(split["train"]),
                         "validation_contexts": len(split["validation"]),
                         "allocated_parameters": sum(p.numel() for p in model.parameters()),
                         "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    write_json(SUITE / "protocol.json", protocol)
    return configs


def collect(configs):
    rows, records, comparisons = [], {}, []
    for name in LABELS:
        path = resolve(configs[name]["output_dir"]) / "evaluation/final_500/bsrt_sample_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        records[name] = frame
        for level, part in frame.groupby("removal_percent"):
            if len(part) != 500 or part.question_id.nunique() != 500:
                continue
            rows.append({"model": name, "removal_percent": level, "questions": 500,
                         **{metric: part[metric].mean() for metric in
                            ["accuracy", "f1", "semantic_similarity", "actual_token_reduction"]}})
    summary = pd.DataFrame(rows)
    summary.to_csv(SUITE / "metrics.csv", index=False)
    rng = np.random.default_rng(20261103)
    reference = records["full"]
    for name in list(LABELS)[1:]:
        if name not in records:
            continue
        paired = reference[reference.removal_percent.eq(40)].merge(
            records[name][records[name].removal_percent.eq(40)],
            on=["context_id", "question_id"], suffixes=("_full", "_variant"), validate="one_to_one")
        if len(paired) != 500:
            continue
        paired["difference"] = paired.f1_full - paired.f1_variant
        clusters = paired.groupby("context_id").difference.agg(["sum", "count"])
        sums, counts = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
        selected = rng.integers(0, len(sums), size=(10000, len(sums)))
        low, high = np.quantile(sums[selected].sum(axis=1)/counts[selected].sum(axis=1), [.025, .975])
        signs = rng.choice([-1, 1], size=(50000, len(sums)))
        difference = paired.difference.mean()
        p = (1 + np.sum(np.abs(signs @ sums / counts.sum()) >= abs(difference)-1e-12))/50001
        comparisons.append({"comparison": "full minus " + name, "mean_f1_difference_40": difference,
                            "ci_low": low, "ci_high": high, "p_raw": p})
    tests = pd.DataFrame(comparisons)
    if len(tests):
        tests["p_holm"] = np.nan
        maximum = 0.
        for rank, idx in enumerate(tests.p_raw.sort_values().index):
            maximum = max(maximum, min(1., (3-rank)*tests.loc[idx, "p_raw"]))
            tests.loc[idx, "p_holm"] = maximum
        tests.to_csv(SUITE / "paired_f1_40.csv", index=False)
    complete = summary.groupby("model").filter(lambda group: len(group) == 10)
    average = complete.groupby("model")[["accuracy", "f1", "semantic_similarity"]].mean().reset_index()
    average.to_csv(SUITE / "across_level_means.csv", index=False)
    status = pd.DataFrame([{"Model": LABELS[name],
                            "Completed 500-question levels": int(summary.model.eq(name).sum()),
                            "Status": "Complete" if summary.model.eq(name).sum() == 10 else "Pending / incomplete"}
                           for name in LABELS])
    lines = ["# Three input ablations of full-reference BSRT", "",
             "## Evaluation status", "", markdown_table(status), "",
             "Results for incomplete variants are not yet available; do not interpret them "
             "as zero performance or draw conclusions before their evaluation completes.", "",
             "Only the contextual BERT vector, learned absolute-position embedding, and budget "
             "embedding are ablated, one at a time. Full-reference A0 and the completed budget "
             "ablation A2 are reused without retraining or changing weights. The two new "
             "variants train from scratch with the same initialization seed, compatible shared "
             "module initial values, splits, 256 training contexts/1,980 questions, 12 validation "
             "contexts/88 questions, eight epochs, answer-span targets, AdamW learning rate "
             "0.0005, weight decay 0.01, gradient clipping 1, and 40% training budget.", "",
             "The contextual-vector ablation removes the entire projected BERT-vector contribution "
             "including projection bias; it does NOT remove the retention Transformer. This "
             "tests the semantic-vector input as a whole, not contextual hidden states versus "
             "static token embeddings. S1-S8, learned positions, and budget remain. The "
             "position ablation removes only the learned absolute-position embedding; S1 "
             "normalized position and BERT positional information remain. The budget ablation "
             "removes only the learned budget input; requested hard token-count selection remains. "
             "No variants remove all structural features or change the supervision objective.", "",
             "Disabled input branches are frozen and omitted including biases, but remain "
             "allocated for checkpoint compatibility. Full-model default outputs are unchanged. "
             "No questions or answers enter inference. Frozen context features and QA predictions "
             "reuse caches. Caching means these timings are not end-to-end cost measurements.", "",
             "Evaluation: identical historical 500 answerable SQuAD v1.1 validation questions, "
             "36 contexts, targets 5-50% in steps of five. Training uses SQuAD 2.0 training data "
             "with article-grouped splits and final context/question exclusions. These historical "
             "test questions have been repeatedly viewed; no pristine-test claim is made. "
             "Checkpoints are selected on validation context-mean F1 only. Seed 20261103. "
             "The primary endpoint is F1 at the common training budget 40%; other budgets "
             "measure transfer. Across-level averages equally weight ten levels and 500 "
             "questions per level. Only complete levels/runs enter their respective tables.", "",
             "Accuracy preserves the existing normalized equality/answer substring OR F1>=0.5 "
             "OR answer semantic cosine>=0.75 criterion. It is not exact match or manually "
             "verified correctness. Semantic similarity uses all-MiniLM-L6-v2. All QA uses "
             "deepset/bert-base-cased-squad2. Token F1, binary accuracy, and similarity are "
             "reported unchanged.", "", "## Across-level means (completed models only)", "",
             markdown_table(average), "", "## All completed levels", "", markdown_table(summary), "",
             "## Paired F1 comparisons at 40%", "",
             "95% pointwise percentile intervals: 10,000 context bootstrap samples retaining "
             "all questions per sampled context, with question-weighted means. Two-sided "
             "context sign-flip tests: 50,000 draws, plus-one Monte Carlo correction, cluster "
             "symmetry assumption, Holm correction across the three planned contrasts, "
             "alpha=0.05. These are not t-tests or simultaneous confidence bands. One training "
             "seed does not quantify training-run variability. No improvement is presumed.", "",
             tests.to_csv(index=False) if len(tests) else "Pending.", "",
             "Source locations and full configurations: protocol.json. Figures: "
             "input_ablation_accuracy.pdf, input_ablation_f1.pdf, "
             "input_ablation_semantic_similarity.pdf (also SVG/PNG). Missing variants "
             "are pending, never zero. No thesis files or previous experiment outputs are modified."]
    (SUITE / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plot(summary)


def plot(frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#c22c42", "#2462b5", "#258354", "#8a58a8"]
    for metric, title in [("accuracy", "Accuracy"), ("f1", "Answer F1"),
                          ("semantic_similarity", "Semantic similarity")]:
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        for (name, label), color in zip(LABELS.items(), colors):
            part = frame[frame.model.eq(name)].sort_values("removal_percent")
            if len(part):
                ax.plot(part.removal_percent, part[metric], marker="o", label=label, color=color,
                        linewidth=2.3 if name == "full" else 1.6, markersize=4)
        ax.set(xlabel="Target token removal (%)", ylabel=title, ylim=(0, 1), xticks=list(range(5, 51, 5)))
        ax.grid(alpha=.2)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.24), ncol=2, fontsize=8.5, frameon=False)
        fig.tight_layout()
        for suffix in ["pdf", "svg", "png"]:
            fig.savefig(SUITE / f"input_ablation_{metric}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    configs = prepare()
    collect(configs)
    if args.prepare_only or args.report_only:
        print(f"Prepared three-input study: {SUITE}", flush=True)
        return
    for name in ["no_contextual_vector", "no_learned_position"]:
        path = resolve(configs[name]["output_dir"]) / "config.json"
        for stage in ["train", "final"]:
            print(f"INPUT ABLATION {name}: {stage}", flush=True)
            subprocess.run([sys.executable, "-u", "-m", "bsrt_neural_compression.run", stage,
                            "--config", str(path)], cwd=ROOT, check=True)
        collect(configs)
    print(f"Completed three-input ablations: {SUITE}", flush=True)


if __name__ == "__main__":
    main()
