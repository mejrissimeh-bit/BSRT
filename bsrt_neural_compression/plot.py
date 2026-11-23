from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from .common import read_json, resolve

LABELS = {"bsrt": "BSRT", "learned_token_policy": "Learned policy", "old_selector": "Method selector",
          "existing_question_oracle": "Question-aware oracle", "context_hindsight_f1": "Context hindsight",
          "original": "Original", "l2": "L2", "attention": "Attention", "random": "Random"}


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_results(config):
    output = resolve(config["output_dir"])
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    history = output / "training_history.jsonl"
    if history.exists():
        rows = pd.DataFrame([json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(rows)+1), rows.validation_context_mean_f1, "o-", color="#a82647")
        ax.set(xlabel="Completed training epoch", ylabel="Validation mean context QA F1", ylim=(0, 1))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        fig.tight_layout()
        for extension in ["svg", "png"]:
            fig.savefig(plots / f"training_validation.{extension}", dpi=160)
        plt.close(fig)
    for partition in ["validation", "untouched", "final_500"]:
        path = output / "evaluation" / partition / "comparison_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if config.get("comparison_methods"):
            frame = frame[frame.method.isin(config["comparison_methods"])].copy()
        if partition == "final_500" and config.get("plot_zero_reference"):
            columns = ["squad_id", "original_accuracy", "original_f1", "original_semantic_similarity"]
            reference = pd.read_csv(resolve(config["final_archive"]) / "standard_methods_sample_results.csv", usecols=columns)
            if (reference.groupby("squad_id")[columns[1:]].nunique() > 1).any().any():
                raise ValueError("Uncompressed reference metrics disagree for a question")
            reference = reference.drop_duplicates("squad_id")
            if len(reference) != 500:
                raise ValueError("Expected the final 500 uncompressed reference outcomes")
            anchor = {metric: reference["original_"+metric].mean() for metric in ["accuracy", "f1", "semantic_similarity"]}
            zero = pd.DataFrame([{**anchor, "method": method, "removal_percent": 0,
                                  "actual_token_reduction": 0.0, "compression_seconds": 0.0}
                                 for method in frame.method.unique()])
            frame = pd.concat([zero, frame], ignore_index=True)
        frame.to_csv(plots / f"{partition}_plot_data.csv", index=False)
        for metric in ["accuracy", "f1", "semantic_similarity"]:
            if metric not in frame:
                continue
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            metric_label = {"f1": "Answer F1", "accuracy": "Accuracy", "semantic_similarity": "Semantic similarity"}[metric]
            if frame.removal_percent.nunique() == 1:
                ax.barh([LABELS.get(m, m) for m in frame.method], frame[metric], color=["#bd2949" if m == "bsrt" else "#337f8c" for m in frame.method])
                for i, value in enumerate(frame[metric]):
                    ax.text(min(value+.012, .95), i, f"{value:.3f}", va="center", fontsize=9)
                ax.invert_yaxis()
                ax.set(xlabel=metric_label, ylabel="Strategy", xlim=(0, 1), title=f"{int(frame.removal_percent.iloc[0])}% token removal")
                ax.grid(axis="x", alpha=.2)
            else:
                for method, rows in frame.groupby("method"):
                    rows = rows.sort_values("removal_percent")
                    styles = {"random": ("#8795a5", ":", "o"), "l2": ("#2265af", "-", "^"),
                              "attention": ("#d98300", "--", "s"), "learned_token_policy": ("#854ac6", "--", "o"),
                              "bsrt": ("#bd2949", "-", "D"), "existing_question_oracle": ("#252a30", "--", "o")}
                    color, line, marker = styles.get(method, (None, "-", "o"))
                    label = "Learned token policy" if method == "learned_token_policy" else LABELS.get(method, method)
                    ax.plot(rows.removal_percent, rows[metric], marker=marker, color=color, linestyle=line, label=label,
                            linewidth=2.8 if method == "bsrt" else 1.7, markersize=4)
                ax.set_xlabel("Target token removal (%)")
                if config.get("plot_zero_reference"):
                    ax.set_xlim(0, 50)
                    ax.set_xticks([0, 10, 20, 30, 40, 50])
                ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.23), ncol=3, frameon=False, fontsize=9)
                ax.set_ylabel(metric_label)
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            fig.tight_layout()
            for extension in config.get("plot_formats", ["svg", "png"]):
                fig.savefig(plots / f"{partition}_{metric}.{extension}", dpi=160)
            plt.close(fig)
    summary = output / "training_summary.json"
    if summary.exists():
        values = read_json(summary)
        lines = ["# BSRT Experiment Results", "", f"- Training contexts: {values['train_contexts']}",
                 f"- Validation contexts: {values['validation_contexts']}",
                 f"- Best validation context-mean F1: {values['best_validation_f1']:.4f}",
                 f"- Selected checkpoint: {values['best_stage']}, epoch {values['best_epoch']}",
                 f"- New QA calls this invocation: {values['new_qa_calls_this_invocation']}",
                 f"- Wall time this invocation: {values['elapsed_seconds_this_invocation']:.1f} seconds", ""]
        for partition in ["validation", "untouched", "final_500"]:
            path = output / "evaluation" / partition / "comparison_summary.csv"
            if path.exists():
                lines += [f"## {partition}", "", markdown_table(pd.read_csv(path)), ""]
        lines += ["## Interpretation limits", "", "Validation determines model selection and is not an unbiased final test. The final 500 are reserved from the new training process but have been viewed repeatedly in previous thesis experiments. Evaluate the additional untouched article split only after freezing the model. Improvements must be established by matched downstream QA results; training loss alone is insufficient."]
        (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
