"""Freeze a validation-selected experiment before the final-500 assessment."""
from __future__ import annotations

import argparse

import pandas as pd

from .common import digest, load_config, read_json, resolve, seed_all, write_json

CONFIGS = ["bsrt_neural_compression/config_pilot.json", "bsrt_neural_compression/config_span_256_40.json"]
STUDY = resolve("bsrt_neural_compression/results/study_40")


def freeze():
    rows = []
    validation_ids = None
    for path in CONFIGS:
        config = load_config(resolve(path))
        output = resolve(config["output_dir"])
        summary = read_json(output / "training_summary.json")
        manifest = read_json(output / "splits.json")
        current_ids = [row["context_id"] for row in manifest["validation"]]
        if validation_ids is not None and validation_ids != current_ids:
            raise ValueError("Candidate validation contexts must match")
        validation_ids = current_ids
        rows.append({"config_path": path, "output_dir": config["output_dir"], **summary})
    chosen = max(rows, key=lambda row: row["best_validation_f1"])
    checkpoint = resolve(chosen["output_dir"]) / "checkpoints" / "best.pt"
    record = {"selection_metric": "mean context QA F1 on identical validation contexts",
              "candidates": rows, "chosen_config": chosen["config_path"],
              "chosen_checkpoint_sha256": digest(checkpoint.read_bytes())}
    STUDY.mkdir(parents=True, exist_ok=True)
    path = STUDY / "frozen_selection.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("The study is frozen. Do not change the selected model using final results.")
    write_json(path, record)
    return load_config(resolve(record["chosen_config"]))


def report():
    from .plot import markdown_table
    selection = read_json(STUDY / "frozen_selection.json")
    config = load_config(resolve(selection["chosen_config"]))
    output = resolve(config["output_dir"])
    final_path = output / "evaluation" / "final_500" / "bsrt_sample_results.csv"
    if final_path.exists():
        final = pd.read_csv(final_path)
        audit = {"question_rows": len(final), "unique_questions": final.question_id.nunique(),
                 "unique_contexts": final.context_id.nunique(),
                 "maximum_compressed_versions_per_context": int(final.groupby("context_id").compressed_context.nunique().max()),
                 "mask_budget_mismatches": int(((final.original_token_count-(final.original_token_count*.4).astype(int)) != final.kept_mask_tokens).sum()),
                 "retokenization_count_changes": int((final.kept_mask_tokens != final.compressed_token_count).sum())}
        write_json(STUDY / "final_audit.json", audit)
    lines = ["# BSRT 40% Pilot Study", "", "## Selection before final evaluation", "",
             "The final 500 QA pairs and all 36 source contexts were excluded from training and validation. Candidate models were selected using identical 12-context SQuAD2 validation data, without consulting final-test rewards.", "",
             "| Experiment | Training contexts | Best validation context-mean F1 | Selected stage |",
             "| --- | ---: | ---: | --- |"]
    for row in selection["candidates"]:
        lines.append(f"| {row['output_dir']} | {row['train_contexts']} | {row['best_validation_f1']:.4f} | {row['best_stage']}, epoch {row['best_epoch']} |")
    lines += ["", f"Frozen model: `{selection['chosen_config']}`.", "",
              "The larger-data experiment varies both the number of training contexts and scheduled training epochs. It is an exploratory comparison rather than an isolated causal ablation of training-set size."]
    for partition in ["validation", "final_500", "untouched"]:
        folder = output / "evaluation" / partition
        path = folder / "comparison_summary.csv"
        if path.exists():
            lines += ["", f"## {partition}", "", markdown_table(pd.read_csv(path)), ""]
        if partition == "final_500" and (folder / "paired_f1_differences.csv").exists():
            lines += ["Paired question-mean F1 differences with 95% context-cluster bootstrap intervals:", "",
                      markdown_table(pd.read_csv(folder / "paired_f1_differences.csv")), ""]
    lines += ["## Search and distillation findings", "",
              "The 32-context experiment completed two multi-question search/distillation rounds. Search rewards are training-only hindsight results. Its validation-best model remained the answer-span checkpoint, so this pilot does not establish that mask distillation or iterative improvement improves generalisation.", "",
              "## Generalisation finding", "",
              "The fixed final-500 cohort shows a substantial BSRT F1 gain over the prior policy (0.7498 versus 0.6306). On the additional 88-question article-held-out partition, F1 is 0.6945 versus 0.6927: effectively tied. BSRT has lower accuracy and semantic similarity than the prior policy on that additional partition. The current evidence therefore supports improvement on the historical final cohort, not a universal improvement across datasets or metrics.", "",
              "## What remains unexecuted", "",
              "The substantial multi-budget training configuration and controlled ablation runner are implemented, but their results must not be claimed until those runs complete. The completed study concerns 40% removal. It does not establish improvements across 5--50% or across training seeds.", "",
              "## Test interpretation", "",
              "The historical 500-pair SQuAD v1.1 validation cohort has been examined in previous thesis work. It is excluded from this new training process but is not a newly untouched benchmark. The additional reserved article partition is reported separately. The frozen QA teacher was previously trained on SQuAD2; this study measures compressor performance, not independent QA model generalisation.", "",
              "## Files", "", f"- Chosen model: `{output / 'checkpoints' / 'best.pt'}`.",
              f"- Final paired predictions: `{output / 'evaluation' / 'final_500' / 'all_comparison_sample_results.csv'}`.",
              f"- Plots: `{output / 'plots'}`.", "- Architecture and losses: `bsrt_neural_compression/TECHNICAL_DESCRIPTION.md`.",
              "- Training, final evaluation, and ablations: `bsrt_neural_compression/README.md`."]
    path = STUDY / "REPORT.md"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "evaluate", "report"])
    args = parser.parse_args()
    if args.command == "report":
        report()
        return
    config = freeze()
    if args.command == "evaluate":
        from .evaluate import final_evaluation, evaluate_development
        from .plot import plot_results
        seed_all(config["seed"], config["threads"])
        final_evaluation(config)
        evaluate_development(config, "untouched")
        plot_results(config)
    report()


if __name__ == "__main__":
    main()
