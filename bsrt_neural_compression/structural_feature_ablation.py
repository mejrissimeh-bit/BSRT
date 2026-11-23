"""Matched S4/S5 input ablations; no feature selection on final-test outcomes."""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys

import numpy as np
import pandas as pd

from .common import ROOT, load_config, read_json, resolve, write_json
from .model import RetentionNetwork
from .plot import markdown_table


SUITE = resolve("bsrt_neural_compression/results/two_structural_feature_ablations")
REFERENCE = resolve("bsrt_neural_compression/results/controlled_ablation/A0")
VARIANTS = {"without_digit_S4": 3, "without_uppercase_S5": 4}
LABELS = {"full": "Full BSRT", "without_digit_S4": "Without digit indicator (S4)",
          "without_uppercase_S5": "Without initial-uppercase indicator (S5)"}
METRICS = ["accuracy", "f1", "semantic_similarity"]


def prepare():
    base = read_json(REFERENCE / "config.json")
    split = read_json(REFERENCE / "splits.json")
    configs = {"full": load_config(REFERENCE / "config.json")}
    for name, index in VARIANTS.items():
        config = copy.deepcopy(base)
        config.update(output_dir=str(SUITE / name), match_reference_initialization=True)
        config.pop("initialize_from", None)
        config["model"]["disabled_structural_features"] = [index]
        folder = resolve(config["output_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "config.json"
        if path.exists() and read_json(path) != config:
            raise ValueError(f"Configuration changed: {name}")
        write_json(path, config)
        configs[name] = load_config(path)
        copied = copy.deepcopy(split)
        copied["config_fingerprint"] = configs[name]["config_fingerprint"]
        write_json(folder / "splits.json", copied)
    protocol = []
    for name, config in configs.items():
        manifest = read_json(resolve(config["output_dir"]) / "splits.json")
        if any(manifest[key] != split[key] for key in ["train", "validation"]):
            raise ValueError("Training/validation data differ")
        network = RetentionNetwork(**config["model"])
        protocol.append({"id": name, "label": LABELS[name], "config": config,
                         "reused_completed_reference": name == "full", "warm_start": False,
                         "disabled_zero_based_structural_index": VARIANTS.get(name),
                         "train_contexts": len(split["train"]),
                         "train_questions": sum(len(row["questions"]) for row in split["train"]),
                         "validation_contexts": len(split["validation"]),
                         "validation_questions": sum(len(row["questions"]) for row in split["validation"]),
                         "allocated_parameters": sum(p.numel() for p in network.parameters()),
                         "nominal_trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad)})
    write_json(SUITE / "protocol.json", protocol)
    return configs


def collect(configs):
    rows, records, tests = [], {}, []
    for name, config in configs.items():
        path = resolve(config["output_dir"]) / "evaluation/final_500/bsrt_sample_results.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        records[name] = frame
        for level, part in frame.groupby("removal_percent"):
            if len(part) != 500 or part.question_id.nunique() != 500:
                continue
            rows.append({"model": name, "removal_percent": level, "questions": 500,
                         **{metric: part[metric].mean() for metric in METRICS + ["actual_token_reduction"]}})
    summary = pd.DataFrame(rows)
    summary.to_csv(SUITE / "metrics.csv", index=False)
    complete = summary.groupby("model").filter(lambda group: len(group) == 10)
    means = complete.groupby("model")[METRICS].mean().reset_index()
    means.to_csv(SUITE / "across_level_means.csv", index=False)
    rng = np.random.default_rng(20261103)
    full = records["full"]
    for name in VARIANTS:
        if name not in records:
            continue
        other = records[name]
        paired = full[full.removal_percent.eq(40)].merge(other[other.removal_percent.eq(40)],
                   on=["context_id", "question_id"], suffixes=("_full", "_variant"), validate="one_to_one")
        if len(paired) != 500:
            continue
        paired["difference"] = paired.f1_full - paired.f1_variant
        clusters = paired.groupby("context_id").difference.agg(["sum", "count"])
        sums, counts = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
        selected = rng.integers(0, len(sums), size=(10000, len(sums)))
        low, high = np.quantile(sums[selected].sum(axis=1)/counts[selected].sum(axis=1), [.025, .975])
        difference = paired.difference.mean()
        signs = rng.choice([-1, 1], size=(50000, len(sums)))
        p = (1 + np.sum(np.abs(signs@sums/counts.sum()) >= abs(difference)-1e-12))/50001
        tests.append({"comparison": "full minus " + name, "mean_f1_difference_40": difference,
                      "ci_low": low, "ci_high": high, "p_raw": p})
    test_frame = pd.DataFrame(tests)
    if len(test_frame):
        test_frame["p_holm"] = np.nan
        maximum = 0.
        for rank, idx in enumerate(test_frame.p_raw.sort_values().index):
            maximum = max(maximum, min(1., (2-rank)*test_frame.loc[idx, "p_raw"]))
            test_frame.loc[idx, "p_holm"] = maximum
        test_frame.to_csv(SUITE / "paired_f1_40.csv", index=False)
    status = pd.DataFrame([{"Model": LABELS[name], "Complete levels": int(summary.model.eq(name).sum()),
                           "Status": "Complete" if summary.model.eq(name).sum() == 10 else "Pending"}
                          for name in LABELS])
    lines = ["# Two individual structural-feature ablations of full-reference BSRT", "",
             "## Status", "", markdown_table(status), "", "## Hypotheses and scope", "",
             "This report contains ONLY two separate ablations: S4 digit indicator and S5 "
             "initial-uppercase indicator, against the unchanged full-reference A0. They were "
             "chosen as plausible cues for numerical facts and named entities, BEFORE observing "
             "these two final results. They are not established as the two most important "
             "features. No search across S1-S8 or selection of only degrading variants is "
             "performed. All outcomes, including null effects and improvements, are retained. "
             "This is an additional exploratory study following previous viewed experiments.", "",
             "## Exact feature definitions and intervention", "",
             "S4 = float(any(c.isdigit() for c in token_string)). S5 = "
             "float(token_string[:1].isupper()). Strings are BERT WordPiece strings: a "
             "continuation prefixed by ## is not initially uppercase under this rule. S5 is "
             "a capitalization cue, not a named-entity detector.", "",
             "The selected feature column is zeroed before the eight-dimensional structural "
             "projection, during both fresh training and evaluation. No other column is "
             "changed. Projection bias, contextual BERT vectors, learned positions, budget "
             "embedding, retention Transformer, and whole-word budget selection remain. "
             "Network dimensions and nominal parameter counts stay unchanged (320,353 "
             "retention parameters); 96 projection weights attached to the masked column "
             "have no data-gradient contribution. Frozen BERT can still encode digit and "
             "capitalization information: this isolates the EXPLICIT feature, not all "
             "possible access to numbers or case.", "",
             "## Matched protocol", "",
             "Full reference predictions are reused without changing weights. Each new "
             "variant starts from fresh initialization (no trained warm start), seed 20261103, "
             "with compatible shared modules initialized identically. The original article-grouped "
             "training/validation manifests are copied exactly: 256 training contexts with "
             "1,980 questions and 12 validation contexts with 88 questions from SQuAD 2.0 "
             "training data. Final evaluation questions and context hashes are excluded. "
             "The frozen encoder/QA model is deepset/bert-base-cased-squad2. The retention "
             "network has width 96, two Transformer layers, four heads, dropout 0.1. All "
             "variants use answer-span-union labels, weighted BCE plus budget penalty 0.1, "
             "AdamW lr 0.0005/weight decay 0.01, gradient clipping 1, one context-budget "
             "update, eight epochs, training budget 40%, and validation context-mean F1 "
             "checkpoint selection. Other removal levels measure transfer, not multi-budget "
             "training. Context features and QA predictions reuse caches; no compression "
             "questions/answers are available at deployment.", "",
             "Evaluation: the same historical 500 answerable SQuAD v1.1 validation questions "
             "from 36 contexts, 5-50% removal in steps of five. Question-weighted means, "
             "not context-weighted means. Across-level means equally weight ten levels, "
             "and are shown only after all ten levels finish. These questions have been "
             "viewed repeatedly during thesis development; they are not an untouched "
             "discovery test. One training seed does not establish seed robustness.", "",
             "Metrics preserve the project definitions: answer-token F1; answer embedding "
             "cosine similarity using all-MiniLM-L6-v2; binary accuracy from normalized "
             "equality/reference substring OR F1 >= 0.5 OR semantic cosine >= 0.75. This "
             "accuracy is not benchmark Exact Match or manually verified factual correctness.", "",
             "## Across-level means (completed variants only)", "", markdown_table(means), "",
             "## All complete removal levels", "", markdown_table(summary), "",
             "## Paired F1 contrasts at 40% removal", "",
             "Differences are full reference minus ablation: positive means lower F1 "
             "without the feature. Primary matched-budget endpoint: F1 at 40%. Pointwise "
             "95% percentile CIs resample 36 contexts 10,000 times, keeping questions within "
             "each cluster and using question-weighted means. Two-sided context sign-flip "
             "tests use 50,000 draws, a plus-one Monte Carlo correction, and assume independent "
             "contexts/sign symmetry under the null. Holm correction covers these two "
             "planned contrasts at alpha 0.05, not the entire history of thesis experiments. "
             "Across-level means and other budgets are descriptive. Single-seed training "
             "uncertainty is not captured.", "",
             test_frame.to_csv(index=False) if len(test_frame) else "Pending; no statistical results invented.", "",
             "## Interpretation"]
    reference_mean = means[means.model.eq("full")].iloc[0]
    for name in VARIANTS:
        result = means[means.model.eq(name)]
        if result.empty:
            lines += ["", f"{LABELS[name]}: evaluation pending; no conclusion yet."]
            continue
        values = result.iloc[0]
        differences = {metric: values[metric]-reference_mean[metric] for metric in METRICS}
        lines += ["", f"{LABELS[name]}: across-level changes (ablation minus reference) are "
                  f"accuracy {differences['accuracy']:+.6f}, F1 {differences['f1']:+.6f}, "
                  f"semantic similarity {differences['semantic_similarity']:+.6f}. "
                  "Negative values indicate descriptive degradation; positive values indicate "
                  "improvement. Statistical inference above concerns 40% F1 only."]
    lines += ["", "A small or null effect does not establish equivalence or prove that a "
              "feature cannot help; redundancy with contextual BERT or other inputs may "
              "limit its incremental benefit. The results do not identify the globally "
              "optimal feature subset. No threshold, QA metric, baseline, oracle, or thesis "
              "figure is changed.", "", "## Sources", "",
              "Configs, exact data manifests, masked indices and parameter counts: protocol.json "
              "and each variant/config.json, splits.json, training_summary.json, checkpoints/.", "",
              "Predictions: each variant/evaluation/final_500/bsrt_sample_results.csv. "
              "Full reference: ../controlled_ablation/A0/evaluation/final_500/bsrt_sample_results.csv. "
              "Tables: metrics.csv, across_level_means.csv, paired_f1_40.csv. "
              "Plots: structural_ablation_accuracy, structural_ablation_f1, "
              "structural_ablation_semantic_similarity, in PDF/SVG/PNG. "
              "Script: ../../structural_feature_ablation.py; masks: ../../model.py. "
              "Only completed observations are reported; missing results are pending, never zero."]
    (SUITE / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    plot(summary)


def plot(frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        for (name, label), color in zip(LABELS.items(), ["#c22c42", "#2462b5", "#258354"]):
            part = frame[frame.model.eq(name)].sort_values("removal_percent")
            if len(part):
                ax.plot(part.removal_percent, part[metric], "o-", color=color, label=label, markersize=4)
        ax.set(xlabel="Target token removal (%)", ylabel={"f1": "Answer F1", "accuracy": "Accuracy",
               "semantic_similarity": "Semantic similarity"}[metric], xticks=list(range(5, 51, 5)), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.2), frameon=False, fontsize=9)
        fig.tight_layout()
        for suffix in ["pdf", "svg", "png"]:
            fig.savefig(SUITE / f"structural_ablation_{metric}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    configs = prepare()
    collect(configs)
    if args.prepare_only or args.report_only:
        print(f"Prepared two-feature study: {SUITE}", flush=True)
        return
    for name in VARIANTS:
        path = resolve(configs[name]["output_dir"]) / "config.json"
        for stage in ["train", "final"]:
            print(f"STRUCTURAL ABLATION {name}: {stage}", flush=True)
            subprocess.run([sys.executable, "-u", "-m", "bsrt_neural_compression.run", stage,
                            "--config", str(path)], cwd=ROOT, check=True)
        collect(configs)
    print(f"Completed two structural-feature ablations: {SUITE}", flush=True)


if __name__ == "__main__":
    main()
