"""Equal-weight K averages on the exact common-context cohort; no new simulations."""
import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONDITION = ["channel", "coding", "removal_percent", "snr_db", "scenario"]
METRICS = ["mean_accuracy", "mean_f1", "mean_semantic_similarity"]
STEM = "semantic_multicast_equal_budget_averaged_k"


def load_averages(output):
    status = json.loads((output/"status.json").read_text(encoding="utf-8"))
    lock = json.loads((output/"run_lock.json").read_text(encoding="utf-8"))
    config = lock["config"]
    if not status["complete"]:
        raise ValueError("Final thesis figure requires a completed experiment")
    columns = ["case_id", "context_id", "group_id", "receiver_count", "removal_percent", "channel",
               "coding", "snr_profile", "budget_regime", "snr_db", "channel_seed", "scenario",
               "total_transmitted_bits", "aggregate_ceiling_bits"]+METRICS
    frame = pd.read_csv(output/"group_results.csv", usecols=columns)
    if len(frame) != status["expected_cases"] or frame.case_id.duplicated().any():
        raise ValueError("Missing or duplicate saved cases")
    if set(frame.budget_regime) != {"equal_aggregate"} or set(frame.snr_profile) != {"heterogeneous"}:
        raise ValueError("Unexpected resource budget or SNR profile")
    if not (frame.total_transmitted_bits <= frame.aggregate_ceiling_bits).all():
        raise ValueError("A saved case violates the aggregate resource ceiling")
    selected = frame[frame.coding.eq("ldpc")].copy()
    counts = [2, 3, 5]
    if set(selected.receiver_count) != set(counts):
        raise ValueError("Expected K=2,3,5")
    contexts = set.intersection(*(set(selected[selected.receiver_count.eq(k)].context_id) for k in counts))
    if not contexts:
        raise ValueError("No common contexts across K")
    selected = selected[selected.context_id.isin(contexts)].copy()
    axes = ["context_id", "receiver_count", "removal_percent", "snr_db", "channel_seed", "channel", "scenario"]
    expected = set(product(sorted(contexts), counts, config["removal_percentages"], config["snr_db_values"],
                           config["channel_seeds"], config["channels"], ["unicast", "multicast"]))
    actual = set(selected[axes].itertuples(index=False, name=None))
    if selected.duplicated(axes).any() or actual != expected or len(selected) != len(expected):
        raise ValueError("Common-context grid is not complete and balanced across K/seeds")
    per_k = selected.groupby(CONDITION+["receiver_count"])[METRICS].mean().reset_index()
    averages = per_k.groupby(CONDITION)[METRICS].mean().reset_index()
    if not per_k.groupby(CONDITION).size().eq(3).all():
        raise ValueError("An averaged point does not include all three K values")
    # On the balanced cohort, group means and the explicit equal-K construction agree.
    direct = selected.groupby(CONDITION)[METRICS].mean().sort_index()
    explicit = averages.set_index(CONDITION)[METRICS].sort_index()
    if not np.allclose(direct, explicit, atol=1e-12):
        raise AssertionError("Incorrect K weighting")
    for metric in METRICS[:2]:
        if not averages[metric].between(0, 1).all():
            raise ValueError("Invalid quality score")
    metadata = {
        "completed_source_cases": len(frame), "selected_cases": len(selected),
        "common_contexts": len(contexts), "context_ids": sorted(contexts), "receiver_counts": counts,
        "channel_seeds": config["channel_seeds"], "removal_percentages": config["removal_percentages"],
        "snr_db_values": config["snr_db_values"], "channels": config["channels"],
        "coding": "ldpc", "configured_rate": config["ldpc_rate"],
        "snr_profile": "heterogeneous", "aggregate_budget": "equal ceiling; measured integer padding retained",
        "averaging": "mean over receivers within group (saved), then equal contexts/seeds within K, then equal mean over K=2,3,5",
        "case_observations_per_point": len(contexts)*len(config["channel_seeds"])*3,
        "uncertainty_intervals": "not estimated; curves are descriptive means",
        "source_group_csv": str(output/"group_results.csv"),
        "source_sha256": hashlib.sha256((output/"group_results.csv").read_bytes()).hexdigest(),
        "plot_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "no_new_channel_or_qa_evaluation": True,
    }
    return averages, per_k, metadata


def make_plot(output):
    averages, per_k, metadata = load_averages(output)
    destination = output/"thesis_figures"
    destination.mkdir(exist_ok=True)
    averages.to_csv(destination/(STEM+"_data.csv"), index=False)
    per_k.to_csv(destination/(STEM+"_per_k.csv"), index=False)
    (destination/(STEM+"_metadata.json")).write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11.5,
                         "axes.titlesize": 13, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 7.4), sharex=True, sharey="row")
    colors = ["#2060ad", "#d14c38", "#008272", "#805ab5"]
    levels = metadata["removal_percentages"]
    for col, channel in enumerate(["awgn", "fading"]):
        for row, (metric, label) in enumerate([("mean_accuracy", "QA accuracy"), ("mean_f1", "Token-level F1")]):
            ax = axes[row, col]
            for color, removal in zip(colors, levels, strict=True):
                for scenario in ["unicast", "multicast"]:
                    part = averages[averages.channel.eq(channel) & averages.removal_percent.eq(removal) & averages.scenario.eq(scenario)].sort_values("snr_db")
                    ax.plot(part.snr_db, part[metric], color=color,
                            linestyle="--" if scenario == "unicast" else "-",
                            marker="o" if scenario == "unicast" else "x", markerfacecolor="none",
                            markersize=5.8 if scenario == "unicast" else 4.5, linewidth=1.65,
                            markeredgewidth=1.15)
            ax.set(ylim=(-.02, 1.02), xlim=(-.4, 15.4))
            ax.set_xticks(metadata["snr_db_values"])
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(alpha=.22, linewidth=.65)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title("AWGN" if channel == "awgn" else "TDL multipath fading", pad=10)
            else:
                ax.set_xlabel(r"Group-center SNR $E_s/N_0$ (dB)", labelpad=7)
    color_handles = [Line2D([], [], color=c, linewidth=2, label=f"{r}%") for c, r in zip(colors, levels, strict=True)]
    fig.legend(handles=color_handles, title="Target token removal", loc="center", bbox_to_anchor=(.52, .115),
               ncol=4, frameon=False, fontsize=10.5, title_fontsize=11)
    style_handles = [Line2D([], [], color="#222222", linestyle="--", marker="o", markerfacecolor="none", label="Unicast"),
                     Line2D([], [], color="#222222", linestyle="-", marker="x", label="Semantic multicast")]
    fig.legend(handles=style_handles, loc="center", bbox_to_anchor=(.52, .035), ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.095, right=.985, bottom=.245, top=.935, hspace=.20, wspace=.17)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(destination/(STEM+"."+ext), dpi=220, bbox_inches="tight", pad_inches=.10)
    plt.close(fig)
    print(json.dumps({"plot": str(destination/(STEM+".pdf")), "common_contexts": metadata["common_contexts"],
                      "selected_cases": metadata["selected_cases"], "case_observations_per_point": metadata["case_observations_per_point"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="semantic_multicast/results/heterogeneous_equal_budget")
    args = parser.parse_args()
    directory = Path(args.results)
    make_plot(directory if directory.is_absolute() else ROOT/directory)
