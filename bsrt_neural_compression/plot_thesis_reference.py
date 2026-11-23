"""Render the full-reference comparison as the thesis's three-panel figure."""
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bsrt_neural_compression.common import digest, read_json, resolve


def main():
    output = Path(__file__).resolve().parent / "results" / "controlled_ablation" / "A0" / "plots"
    data = pd.read_csv(output / "final_500_plot_data.csv")
    styles = [
        ("random", "Random", "#82909e", ":", "o"),
        ("attention", "Attention", "#c77c00", "--", "s"),
        ("l2", "L2", "#2462b5", "-", "^"),
        ("bsrt", "BSRT", "#c22c42", "-", "D"),
        ("existing_question_oracle", "Question-aware oracle", "#222222", "--", "v"),
    ]
    hybrid_root = resolve("caveman_l2_final_500/results")
    hybrid_path = hybrid_root / "sample_results.csv"
    if hybrid_path.exists():
        from caveman_l2_final_500.run import validate_results
        lock = read_json(hybrid_root / "run_lock.json")
        if digest((output / "final_500_plot_data.csv").read_bytes()) != lock["baseline_plot_data_sha256"]:
            raise ValueError("The baseline figure data changed since the matched Caveman evaluation")
        hybrid = pd.read_csv(hybrid_path)
        validate_results(hybrid, lock["config"], lock["question_ids"], complete=True)
        summary = hybrid.groupby(["method", "removal_percent"])[[
            "accuracy", "f1", "semantic_similarity", "actual_token_reduction", "compression_seconds"]].mean().reset_index()
        # Show Caveman only on the shared target grid above its base reduction.
        summary = summary[summary.removal_percent.isin([30, 35, 40, 45, 50])]
        data = pd.concat([data, summary], ignore_index=True)
        styles.insert(3, ("caveman_l2", "Caveman+L2", "#159947", "-.", "s"))
    data.to_csv(output / "final_adaptive_quality_plot_data.csv", index=False)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.2))
    for ax, metric, label in zip(axes, ["accuracy", "f1", "semantic_similarity"],
                                  ["Accuracy", "Answer F1", "Semantic similarity"]):
        for method, name, color, line, marker in styles:
            rows = data[data.method.eq(method)].sort_values("removal_percent")
            if method == "caveman_l2":
                if rows.removal_percent.tolist() != [30, 35, 40, 45, 50]:
                    raise ValueError("Expected Caveman+L2 targets 30%, 35%, 40%, 45%, 50%")
            elif len(rows) != 11:
                raise ValueError(f"Expected 0% anchor and ten levels for {method}")
            ax.plot(rows.removal_percent, rows[metric], label=name,
                    color=color, linestyle=line, marker=marker, markersize=3.5,
                    linewidth=2.2 if method == "bsrt" else 1.5,
                    zorder=5 if method == "bsrt" else 2)
        ax.set(xlim=(0, 50), xticks=list(range(0, 51, 5)), ylabel=label,
               xlabel="Target token removal (%)")
        ax.set_ylim({"accuracy": (.4, 1), "f1": (.35, 1), "semantic_similarity": (.6, 1)}[metric])
        ax.grid(alpha=.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=3, frameon=False, fontsize=9, columnspacing=1.4)
    fig.subplots_adjust(left=.12, right=.98, bottom=.06, top=.925, hspace=.39)
    for extension in ["pdf", "svg", "png"]:
        path = output / f"final_adaptive_quality.{extension}"
        if path.exists():
            archive = output / "archive"
            archive.mkdir(exist_ok=True)
            backup = archive / f"final_adaptive_quality_{digest(path.read_bytes())[:12]}.{extension}"
            if not backup.exists():
                shutil.copyfile(path, backup)
        fig.savefig(path, dpi=180, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
