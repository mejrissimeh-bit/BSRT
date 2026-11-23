"""Condition-specific figures; no mixing compression and aggregate-budget comparisons."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
import numpy as np
import pandas as pd

from bsrt_neural_compression.common import read_json, resolve


COLORS = ["#1261a0", "#c13a31", "#008b75", "#8b56ae", "#bf8500"]


def finish(fig, path, title):
    fig.suptitle(title, fontsize=15)
    for ax in fig.axes:
        ax.grid(alpha=.22)
        ax.tick_params(labelsize=10)
        if not isinstance(ax.xaxis.get_major_locator(), FixedLocator):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout(rect=(0, 0, 1, .95), pad=1.6)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(path.with_suffix("."+ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_results(config):
    output = resolve(config["output_dir"])
    status = read_json(output/"status.json")
    if not status["complete"]:
        raise ValueError("Incomplete run: plots are not published as complete results")
    summary = pd.read_csv(output/"summary.csv")
    paired = pd.read_csv(output/"paired_comparison.csv")
    raw_groups = pd.read_csv(output/"group_results.csv")
    receivers = pd.read_csv(output/"receiver_results.csv")
    folder = output/"plots"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
                         "legend.fontsize": 9, "axes.titlesize": 12, "svg.fonttype": "none"})
    for (regime, channel, coding), part in summary.groupby(["budget_regime", "channel", "coding"]):
        stem = f"{regime}_{channel}_{coding}"
        label = f"{'Equal message budgets' if regime == 'equal_per_representation' else 'Equal aggregate budget'}; "
        label += f"{'AWGN' if channel == 'awgn' else 'TDL fading'}, {coding.upper()}"
        levels = sorted(part.removal_percent.unique())
        resource_data = raw_groups[raw_groups.budget_regime.eq(regime) & raw_groups.channel.eq(channel) & raw_groups.coding.eq(coding)]
        context_sets = [set(s.context_id) for _, s in resource_data.groupby("receiver_count")]
        common_contexts = set.intersection(*context_sets)
        resource_data = resource_data[resource_data.context_id.isin(common_contexts)]
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, removal in enumerate(levels):
            for scenario, style in [("unicast", "--"), ("multicast", "-")]:
                data = resource_data[resource_data.removal_percent.eq(removal) & resource_data.scenario.eq(scenario)].groupby("receiver_count").total_qpsk_symbols.mean()
                ax.plot(data.index, data.values, linestyle=style, marker="o" if scenario == "unicast" else "x",
                        markerfacecolor="none", markersize=8 if scenario == "unicast" else 6,
                        color=COLORS[i % len(COLORS)],
                        label=f"{removal}% / {scenario}")
        ax.set(xlabel="Number of receivers K", ylabel="Aggregate transmitted QPSK symbols")
        ax.set_title(f"Matched contexts across K: {len(common_contexts)}")
        ax.set_xticks(sorted(part.receiver_count.unique()))
        ax.legend(ncol=2)
        finish(fig, folder/(stem+"_resources"), "Transmission resources: "+label)

        paired_part = paired[paired.budget_regime.eq(regime) & paired.channel.eq(channel) & paired.coding.eq(coding)]
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, removal in enumerate(levels):
            data = paired_part[paired_part.removal_percent.eq(removal)].groupby("receiver_count").resource_savings.mean()
            ax.plot(data.index, 100*data.values, "-o", color=COLORS[i % len(COLORS)], label=f"{removal}% removal")
        ax.set(xlabel="Number of receivers K", ylabel="Multicast resource savings versus unicast (%)")
        ax.set_xticks(sorted(part.receiver_count.unique()))
        ax.legend()
        finish(fig, folder/(stem+"_savings"), "Resource savings: "+label)

        for k, sub in part.groupby("receiver_count"):
            profiles = [p for p in ["homogeneous", "heterogeneous"] if p in set(sub.snr_profile)]
            fig, axes = plt.subplots(2, len(profiles), figsize=(6*len(profiles), 9), squeeze=False)
            for col, profile in enumerate(profiles):
                for row, metric in enumerate(["mean_accuracy", "mean_f1"]):
                    ax = axes[row, col]
                    for i, removal in enumerate(levels):
                        for scenario, style in [("unicast", "--"), ("multicast", "-")]:
                            data = sub[sub.snr_profile.eq(profile) & sub.removal_percent.eq(removal) & sub.scenario.eq(scenario)].sort_values("snr_db")
                            ax.plot(data.snr_db, data[metric], linestyle=style,
                                    marker="o" if scenario == "unicast" else "x", markerfacecolor="none",
                                    markersize=8 if scenario == "unicast" else 6, color=COLORS[i % len(COLORS)],
                                    label=f"{removal}% / {scenario}")
                    ax.set(xlabel="Group-center SNR, Es/N0 (dB)", ylabel="QA accuracy" if row == 0 else "Token F1", ylim=(-.03, 1.03))
                    ax.set_xticks(sorted(sub.snr_db.unique()))
                    ax.set_title(profile.capitalize())
                    if row == 0 and col == 0:
                        ax.legend(ncol=2, loc="lower right")
            finish(fig, folder/(stem+f"_K{k}_quality_snr"), f"QA performance, K={k}: "+label)

            fig, axes = plt.subplots(1, len(profiles), figsize=(6*len(profiles), 6), squeeze=False)
            chosen_snr = sorted(sub.snr_db.unique())
            for col, profile in enumerate(profiles):
                ax = axes[0, col]
                for i, snr in enumerate(chosen_snr):
                    for scenario, style in [("unicast", "--"), ("multicast", "-")]:
                        data = sub[sub.snr_profile.eq(profile) & sub.snr_db.eq(snr) & sub.scenario.eq(scenario)].sort_values("total_qpsk_symbols")
                        ax.plot(data.total_qpsk_symbols, data.mean_accuracy, linestyle=style,
                                marker="o" if scenario == "unicast" else "x", markerfacecolor="none",
                                markersize=8 if scenario == "unicast" else 6, color=COLORS[i % len(COLORS)],
                                label=f"{snr:g} dB / {scenario}")
                ax.set(xlabel="Aggregate transmitted QPSK symbols", ylabel="QA accuracy", ylim=(-.03, 1.03), title=profile.capitalize())
                ax.legend(ncol=2)
            finish(fig, folder/(stem+f"_K{k}_quality_resources"), f"QA/resource trade-off, K={k}: "+label)

            data = receivers[receivers.budget_regime.eq(regime) & receivers.channel.eq(channel) &
                             receivers.coding.eq(coding) & receivers.receiver_count.eq(k) & receivers.snr_profile.eq("heterogeneous")]
            if data.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            for i, scenario in enumerate(["unicast", "multicast"]):
                curve = data[data.scenario.eq(scenario)].groupby("receiver_snr_db")
                for ax, metric, ylabel in [(axes[0], "accuracy", "QA accuracy"), (axes[1], "decode_success", "Bit-perfect frame decoding fraction")]:
                    mean = curve[metric].mean()
                    ax.plot(mean.index, mean.values, "-o", label=scenario, color=COLORS[i])
                    ax.set(xlabel="Actual receiver SNR, Es/N0 (dB)", ylabel=ylabel, ylim=(-.03, 1.03))
                    ax.legend()
            finish(fig, folder/(stem+f"_K{k}_heterogeneous"), f"Heterogeneous receivers, K={k} (removal levels pooled): "+label)
    print(f"Plots saved: {folder}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="semantic_multicast/config.json")
    args = parser.parse_args()
    plot_results(read_json(resolve(args.config)))
