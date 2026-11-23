"""Matched original unicast versus 40%-removal aligned coded multicast."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from bsrt_neural_compression.common import read_json, resolve


def plot(output_dir):
    output = resolve(output_dir)
    status, plan = read_json(output/"status.json"), read_json(output/"plan.json")
    if not status["complete"]:
        raise ValueError("Wait until all paired cases are complete")
    data = pd.read_csv(output/"group_results.csv")
    keys = ["context_id", "receiver_count", "channel", "snr_db", "channel_seed"]
    if len(data) != plan["expected_cases"] or data.duplicated(keys+["scenario"]).any():
        raise ValueError("Missing or duplicate cases")
    if not data.groupby(keys).scenario.nunique().eq(2).all():
        raise ValueError("Missing paired baseline")
    if not data[data.scenario.eq("unicast")].removal_percent.eq(0).all():
        raise ValueError("Unicast must be uncompressed")
    if not data[data.scenario.eq("multicast")].removal_percent.eq(40).all():
        raise ValueError("Multicast must have 40% target removal")
    if not data.total_transmitted_bits.eq(data.budget_bits).all():
        raise ValueError("Both strategies must use exactly the same aggregate budget")
    if not data.groupby(["receiver_count", "channel", "snr_db", "scenario"]).size().nunique() == 1:
        raise ValueError("Unequal context/seed counts")
    if not data.groupby(keys).budget_bits.nunique().eq(1).all():
        raise ValueError("Budget differs within a comparison")
    mean = pd.read_csv(output/"summary_averaged_k.csv")
    if not data.groupby(["channel", "snr_db", "scenario"]).receiver_count.nunique().eq(3).all():
        raise ValueError("Missing K value")
    folder = output/"plots"
    folder.mkdir(exist_ok=True)
    stem = "original_unicast_vs_aligned40_budgeted_multicast"
    mean.to_csv(folder/(stem+"_data.csv"), index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11.5,
                         "axes.titlesize": 13, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, panels = plt.subplots(2, 2, figsize=(8.5, 7), sharex=True, sharey="row")
    for col, channel in enumerate(["awgn", "fading"]):
        for row, (metric, label) in enumerate([("mean_accuracy", "QA accuracy"), ("mean_f1", "Token-level F1")]):
            ax = panels[row, col]
            for scenario, color, style, marker in [("unicast", "#2060ad", "--", "o"), ("multicast", "#ce442f", "-", "x")]:
                part = mean[mean.channel.eq(channel) & mean.scenario.eq(scenario)].sort_values("snr_db")
                ax.plot(part.snr_db, part[metric], color=color, linestyle=style, marker=marker,
                        markerfacecolor="none", markersize=5.5, linewidth=1.8)
            ax.set_ylim(-.02, 1.02)
            ax.set_xticks(sorted(data.snr_db.unique()))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(alpha=.22, linewidth=.65)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title("AWGN" if channel == "awgn" else "TDL multipath fading", pad=10)
            else:
                ax.set_xlabel(r"Group-center SNR $E_s/N_0$ (dB)", labelpad=7)
    handles = [Line2D([], [], color="#2060ad", linestyle="--", marker="o", markerfacecolor="none",
                      label="Original unicast, no FEC"),
               Line2D([], [], color="#ce442f", marker="x",
                      label="40% removal: aligned multicast + LDPC / repetition")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, .047), frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=.10, right=.98, bottom=.19, top=.94, hspace=.23, wspace=.17)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(folder/(stem+"."+ext), dpi=220, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)
    print(f"Comparison plots: {folder/stem}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="multicast_coding_budget/results/compressed40")
    plot(parser.parse_args().output)
