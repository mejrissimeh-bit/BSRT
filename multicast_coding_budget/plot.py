"""Only the requested original-uncoded-unicast versus original-coded-multicast curves."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from bsrt_neural_compression.common import read_json, resolve


STEM = "original_unicast_vs_budgeted_multicast_averaged_k"


def plot_results(config):
    output = resolve(config["output_dir"])
    status, plan = read_json(output/"status.json"), read_json(output/"plan.json")
    if not status["complete"]:
        raise ValueError("Cannot publish incomplete evaluation as final")
    data = pd.read_csv(output/"group_results.csv")
    axes = ["context_id", "receiver_count", "channel", "snr_db", "channel_seed", "scenario"]
    if data.duplicated(axes).any() or len(data) != plan["expected_cases"]:
        raise ValueError("Missing or duplicate paired cases")
    if not data.removal_percent.eq(0).all() or not (data.total_transmitted_bits <= data.budget_bits).all():
        raise ValueError("Not an uncompressed budget-constrained evaluation")
    counts = data.groupby(["receiver_count", "channel", "snr_db", "scenario"]).size()
    if counts.nunique() != 1:
        raise ValueError("Unbalanced context/seed/K averaging")
    per_k = pd.read_csv(output/"summary_per_k.csv")
    averaged = pd.read_csv(output/"summary_averaged_k.csv")
    if not per_k.groupby(["channel", "snr_db", "scenario"]).size().eq(3).all():
        raise ValueError("Missing group size in final means")
    folder = output/"plots"
    folder.mkdir(exist_ok=True)
    averaged.to_csv(folder/(STEM+"_data.csv"), index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11.5,
                         "axes.titlesize": 13, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, panels = plt.subplots(2, 2, figsize=(8.1, 6.7), sharex=True, sharey="row")
    for col, channel in enumerate(["awgn", "fading"]):
        for row, (metric, label) in enumerate([("mean_accuracy", "QA accuracy"), ("mean_f1", "Token-level F1")]):
            ax = panels[row, col]
            for scenario, color, style, marker in [("unicast", "#2060ad", "--", "o"), ("multicast", "#ce442f", "-", "x")]:
                part = averaged[averaged.channel.eq(channel) & averaged.scenario.eq(scenario)].sort_values("snr_db")
                ax.plot(part.snr_db, part[metric], color=color, linestyle=style, marker=marker,
                        markersize=5.5, markerfacecolor="none", linewidth=1.8)
            ax.set_ylim(-.02, 1.02)
            ax.set_xticks(config["snr_db_values"])
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(alpha=.22, linewidth=.65)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title("AWGN" if channel == "awgn" else "TDL multipath fading", pad=10)
            else:
                ax.set_xlabel(r"Group-center SNR $E_s/N_0$ (dB)", labelpad=7)
    handles = [Line2D([], [], color="#2060ad", linestyle="--", marker="o", markerfacecolor="none", label="Original unicast, no FEC"),
               Line2D([], [], color="#ce442f", linestyle="-", marker="x", label="Original multicast + LDPC")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, .045), ncol=1, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.095, right=.985, bottom=.18, top=.935, hspace=.22, wspace=.17)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(folder/(STEM+"."+ext), dpi=220, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)
    print(f"Comparison plot saved: {folder/(STEM+'.pdf')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="multicast_coding_budget/config.json")
    args = parser.parse_args()
    plot_results(read_json(resolve(args.config)))
