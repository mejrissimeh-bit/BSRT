"""Paired subset analysis: exclude singleton-cluster receivers from BOTH curves."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from bsrt_neural_compression.common import read_json, write_json
from .transport import budget_for_context


HERE = Path(__file__).resolve().parent
SOURCE = HERE/"results/full"
OUTPUT = HERE/"results/non_singleton_receivers"
STEM = "unicast_vs_multicast_non_singleton_receivers"
METRICS = ["accuracy", "f1", "semantic_similarity", "information_ber", "decode_success"]


def prepare():
    if not read_json(SOURCE/"status.json")["complete"] or not read_json(SOURCE/"audit.json")["passed"]:
        raise ValueError("The full paired source experiment must be complete and audited")
    lock = read_json(SOURCE/"run_lock.json")
    eligible = []
    for group in lock["groups"]:
        budget = budget_for_context(group["original_tokens"])
        for cluster in group["clusters"]:
            size = len(cluster["members"])
            if size < 2:
                continue
            for member in cluster["members"]:
                eligible.append({"context_id": group["context_id"],
                    "receiver_id": group["questions"][member]["id"],
                    "eligible_cluster_id": cluster["id"], "cluster_size": size,
                    "original_group_receiver_count": group["k"],
                    "per_receiver_budget_bits": budget["per_receiver_budget_bits"]})
    cohort = pd.DataFrame(eligible)
    keys = ["context_id", "receiver_id"]
    if cohort.duplicated(keys).any() or len(cohort) != 416:
        raise ValueError("Unexpected non-singleton cohort")
    columns = keys+["channel", "snr_db", "channel_seed", "scenario", "receiver_snr_db", "physical_seed",
                    "coding", "payload_bits", "frame_transmitted_bits", "information_errors"]+METRICS
    data = pd.read_csv(SOURCE/"receiver_results.csv", usecols=columns,
                       dtype={"context_id": str, "receiver_id": str})
    data = data.merge(cohort, on=keys, how="inner", validate="many_to_one")
    conditions = ["channel", "snr_db", "channel_seed", "scenario"]
    if not data.groupby(conditions).size().eq(len(cohort)).all():
        raise ValueError("A condition does not contain every eligible question")
    config = lock["config"]
    expected = len(cohort)*len(config["channels"])*len(config["snr_db_values"])*len(config["channel_seeds"])*2
    if len(data) != expected or data.duplicated(keys+conditions).any():
        raise ValueError("Missing or duplicated receiver outcomes")
    pair_keys = keys+["channel", "snr_db", "channel_seed"]
    unicast = data[data.scenario.eq("unicast")].set_index(pair_keys).sort_index()
    multicast = data[data.scenario.eq("multicast")].set_index(pair_keys).sort_index()
    if not unicast.index.equals(multicast.index):
        raise ValueError("Unicast and multicast do not contain the same paired questions")
    for field in ["physical_seed", "receiver_snr_db", "payload_bits"]:
        np.testing.assert_array_equal(unicast[field], multicast[field])
    if not unicast.coding.eq("uncoded").all() or not multicast.coding.eq("ldpc").all():
        raise ValueError("Singleton exclusion does not match the recorded coding assignments")
    allocations = read_json(SOURCE/"budget_allocations.json")
    cluster_budget = sum(a["budget_bits"] for a in allocations if a["members"] > 1)
    individual_budget = int(cohort.per_receiver_budget_bits.sum())
    if cluster_budget != individual_budget:
        raise ValueError("Retained clusters do not have equal aggregate budgets across scenarios")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(OUTPUT/"included_receivers.csv", index=False)
    data.to_csv(OUTPUT/"receiver_results.csv", index=False)
    per_seed = data.groupby(conditions)[METRICS].mean().reset_index()
    per_seed.to_csv(OUTPUT/"summary_per_seed.csv", index=False)
    mean = per_seed.groupby(["channel", "snr_db", "scenario"])[METRICS].mean().reset_index()
    mean.to_csv(OUTPUT/"summary_all_questions.csv", index=False)
    clean = pd.read_csv(SOURCE/"clean_receiver_results.csv",
                        dtype={"context_id": str, "receiver_id": str})
    clean = clean.merge(cohort[keys], on=keys, how="inner", validate="many_to_one")
    if not clean.groupby("scenario").size().eq(len(cohort)).all():
        raise ValueError("Clean reference differs from the plotted subset")
    clean.groupby("scenario")[["accuracy", "f1", "semantic_similarity"]].mean().reset_index().to_csv(
        OUTPUT/"clean_summary.csv", index=False)
    metadata = {"analysis": "Paired descriptive subset of the existing final run; no rerun or retraining",
                "selection": "Include receivers only when their original task-alignment cluster has at least two members",
                "questions": len(cohort), "excluded_singleton_questions": 500-len(cohort),
                "contexts": int(cohort.context_id.nunique()),
                "clusters": int(cohort.drop_duplicates(["context_id", "eligible_cluster_id"]).shape[0]),
                "receiver_outcomes": len(data), "outcomes_per_curve_point": len(cohort)*len(config["channel_seeds"]),
                "seeds": config["channel_seeds"], "snr_db_values": config["snr_db_values"],
                "paired_snr_seeds_payload_length_verified": True,
                "aggregate_budget_per_strategy_per_channel_snr_seed_bits": individual_budget,
                "averaging": "Equal weight to each eligible question, then average the three seeds",
                "snr_assignment": "Unchanged original full-context-group receiver offsets and physical seeds; not reassigned after filtering",
                "limit": "This selected 416-question cohort must not be described as all 500 questions; no significance is claimed",
                "legend_labels": ["Unicast", "Multicast"],
                "source": str(SOURCE),
                "source_run_lock_sha256": hashlib.sha256((SOURCE/"run_lock.json").read_bytes()).hexdigest(),
                "source_receiver_results_sha256": hashlib.sha256((SOURCE/"receiver_results.csv").read_bytes()).hexdigest()}
    write_json(OUTPUT/"subset_metadata.json", metadata)
    (OUTPUT/"README.md").write_text(
        "# Unicast versus multicast: non-singleton receivers\n\n"
        f"This plot includes the same {len(cohort)} questions from {metadata['contexts']} contexts "
        f"and {metadata['clusters']} clusters in both strategies. It excludes all 84 receivers "
        "belonging to singleton task-alignment clusters. Selection depends only on saved cluster "
        "membership, not QA performance or SNR.\n\n"
        "All data are reused from the completed final experiment. Original receiver SNRs, seeds, "
        "messages, protection and bit allocations are unchanged; no channel, compression or QA "
        "run was repeated. Means give equal weight to each retained question, then to the three "
        "seeds. The omitted singleton budgets are removed from both sides of the subset budget ledger.\n\n"
        "Legends say only Unicast and Multicast. In the underlying experiment, unicast is "
        "individually compressed without FEC; multicast is task-aligned with LDPC and repetition "
        "where needed. This shorter legend does not change that methodological distinction.\n\n"
        "These are subgroup results, not the original 500-question population or a new experiment "
        "in which receiver SNRs were reassigned after exclusions. No significance is claimed. "
        "The original full-cohort plots and report remain unchanged.\n", encoding="utf-8")
    return mean, metadata


def plot(mean):
    folder = OUTPUT/"plots"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11.5,
                         "axes.titlesize": 13, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "path.simplify": False})
    fig, panels = plt.subplots(2, 2, figsize=(8.5, 7), sharex=True, sharey="row")
    for col, channel in enumerate(["awgn", "fading"]):
        for row, (metric, label) in enumerate([("accuracy", "QA accuracy"), ("f1", "Token-level F1")]):
            ax = panels[row, col]
            for scenario, color, style, marker in [("unicast", "#2060ad", "--", "o"), ("multicast", "#ce442f", "-", "x")]:
                part = mean[mean.channel.eq(channel) & mean.scenario.eq(scenario)].sort_values("snr_db")
                ax.plot(part.snr_db, part[metric], color=color, linestyle=style, marker=marker,
                        markerfacecolor="none", markersize=5.5, linewidth=1.8)
            ax.set_ylim(-.02, 1.02)
            ax.set_xticks(sorted(mean.snr_db.unique()))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.grid(alpha=.22, linewidth=.65)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title("AWGN" if channel == "awgn" else "TDL multipath fading", pad=10)
            else:
                ax.set_xlabel(r"Group-center SNR $E_s/N_0$ (dB)", labelpad=7)
    handles = [Line2D([], [], color="#2060ad", linestyle="--", marker="o", markerfacecolor="none", label="Unicast"),
               Line2D([], [], color="#ce442f", marker="x", label="Multicast")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.53, .055), ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=.10, right=.98, bottom=.17, top=.94, hspace=.23, wspace=.17)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(folder/(STEM+"."+ext), dpi=220, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)
    mean.to_csv(folder/(STEM+"_data.csv"), index=False)
    print(folder/(STEM+".pdf"))


if __name__ == "__main__":
    mean, metadata = prepare()
    print(json.dumps(metadata, indent=2))
    plot(mean)
