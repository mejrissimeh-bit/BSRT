"""Question-weighted comparison for the complete paired cohort."""
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
    plan = read_json(output/"plan.json")
    if not read_json(output/"status.json")["complete"]:
        raise ValueError("Cannot plot incomplete results as final")
    data = pd.read_csv(output/"receiver_results.csv", dtype={"receiver_id": str})
    counts = data.groupby(["channel", "snr_db", "channel_seed", "scenario"]).receiver_id.nunique()
    if not counts.eq(plan["questions"]).all() or len(data) != plan["expected_receiver_outcomes"]:
        raise ValueError("Missing question-level results")
    keys = ["receiver_id", "channel", "snr_db", "channel_seed", "scenario"]
    if data.duplicated(keys).any():
        raise ValueError("Repeated question outcomes")
    per_seed = data.groupby(["channel", "snr_db", "channel_seed", "scenario"])[["accuracy", "f1"]].mean()
    mean = per_seed.groupby(["channel", "snr_db", "scenario"]).mean().reset_index()
    folder = output/"plots"
    folder.mkdir(exist_ok=True)
    stem = "task_aligned_multicast_vs_individual_unicast_40_percent"
    mean.to_csv(folder/(stem+"_data.csv"), index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11.5,
                         "axes.titlesize": 13, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, panels = plt.subplots(2, 2, figsize=(8.5, 7), sharex=True, sharey="row")
    for col, channel in enumerate(["awgn", "fading"]):
        for row, (metric, label) in enumerate([("accuracy", "QA accuracy"), ("f1", "Token-level F1")]):
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
                      label="Individual compressed unicast, no FEC"),
               Line2D([], [], color="#ce442f", marker="x",
                      label="Task-aligned compressed multicast, budget-funded FEC")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, .047), frameon=False, fontsize=10.5)
    fig.subplots_adjust(left=.10, right=.98, bottom=.19, top=.94, hspace=.23, wspace=.17)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(folder/(stem+"."+ext), dpi=220, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)
    print(f"Final comparison plots: {folder/stem}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="task_aligned_multicast_500/results/full")
    plot(parser.parse_args().output)
