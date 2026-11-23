"""Pointwise paired context-bootstrap CIs for the saved 416-question comparison."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/non_singleton_receivers"
OUT = SOURCE / "confidence_intervals"
REPLICATES = 50000
BOOTSTRAP_SEED = 20261106


def main():
    OUT.mkdir(exist_ok=True)
    path = SOURCE / "receiver_results.csv"
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = json.loads((SOURCE / "subset_metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(path, dtype={"context_id": str, "receiver_id": str})
    assert len(frame) == 29952 and frame.accuracy.isin([0, 1]).all()
    keys = ["context_id", "receiver_id", "channel", "snr_db", "channel_seed"]
    assert not frame.duplicated(keys+["scenario"]).any()
    paired = frame.pivot(index=keys, columns="scenario", values="accuracy")
    assert set(paired.columns) == {"multicast", "unicast"} and not paired.isna().any().any()
    paired["difference"] = paired.multicast-paired.unicast
    paired = paired.reset_index()
    assert paired.groupby(keys[:-1]).size().eq(3).all()
    assert sorted(paired.channel_seed.unique()) == metadata["seeds"]
    questions = paired.groupby(keys[:-1])[["multicast", "unicast", "difference"]].mean().reset_index()
    context_ids = sorted(questions.context_id.unique())
    assert len(context_ids) == 30
    cohort = frame[["context_id", "receiver_id"]].drop_duplicates()
    assert len(cohort) == 416
    sizes = cohort.groupby("context_id").size().reindex(context_ids).to_numpy()
    # A context draw retains every eligible question, both strategies and all seeds.
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(context_ids), size=(REPLICATES, len(context_ids)))
    denominators = sizes[draws].sum(axis=1)
    rows, context_rows = [], []
    for (channel, snr), part in questions.groupby(["channel", "snr_db"]):
        assert len(part) == 416 and part.context_id.nunique() == 30
        group = part.groupby("context_id").agg(
            difference_sum=("difference", "sum"), multicast_sum=("multicast", "sum"),
            unicast_sum=("unicast", "sum"), questions=("receiver_id", "size")).reindex(context_ids)
        np.testing.assert_array_equal(group.questions, sizes)
        sums = group.difference_sum.to_numpy()
        bootstrap = sums[draws].sum(axis=1)/denominators
        low, high = np.quantile(bootstrap, [.025, .975], method="linear")
        estimate = sums.sum()/sizes.sum()
        np.testing.assert_allclose(estimate, part.multicast.mean()-part.unicast.mean(), atol=1e-14, rtol=0)
        rows.append({"channel": channel, "group_center_snr_db": snr, "questions": 416, "contexts": 30,
                     "channel_seeds": 3, "bootstrap_replicates": REPLICATES,
                     "unicast_accuracy": part.unicast.mean(), "multicast_accuracy": part.multicast.mean(),
                     "difference_pp": 100*estimate, "ci95_low_pp": 100*low, "ci95_high_pp": 100*high,
                     "pointwise_ci_excludes_zero": bool(low > 0 or high < 0)})
        context_rows.append(group.assign(context_id=context_ids, channel=channel, group_center_snr_db=snr))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "paired_accuracy_difference_ci95.csv", index=False)
    pd.concat(context_rows, ignore_index=True).to_csv(OUT / "context_contributions.csv", index=False)
    settings = {"source": str(path), "source_sha256": source_hash, "replicates": REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED, "confidence_level": .95,
                "method": "paired context-clustered percentile bootstrap",
                "estimand": "question-weighted multicast minus unicast accuracy, averaged over three fixed channel seeds",
                "resampling": "Draw 30 contexts with replacement, retaining every question, both strategies and all three seeds in each selected context",
                "denominator": "sum of eligible question counts in the resampled contexts, including repeated contexts",
                "interval_scope": "pointwise; not simultaneous across SNR/channel comparisons",
                "channel_seed_scope": "conditional on the three recorded channel seeds; no channel-seed resampling",
                "correctness_rule": "original historical OR/equality/containment rule; neither AND-rescored nor manually corrected labels",
                "p_values_computed": False}
    (OUT / "bootstrap_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    lines = ["# Accuracy-difference confidence intervals: latest multicast versus unicast", "",
             "Cohort: 416 questions / 30 contexts / 109 non-singleton clusters. Both strategies use the same questions and three seeds (42, 123, 2024).", "",
             f"Method: paired context-clustered percentile bootstrap with {REPLICATES:,} replicates and bootstrap seed {BOOTSTRAP_SEED}. Each replicate samples 30 contexts with replacement. Both strategies, every eligible question and all three channel seeds are retained within each sampled context.",
             "The resampled accuracy difference is the sum of per-context question-level differences divided by the sum of sampled question counts. Thus unequal context sizes retain the original question-weighted estimand, rather than changing to an equal-context mean.",
             "Intervals are pointwise 95% percentile intervals, conditional on the observed channel seeds. They are not simultaneous familywise intervals. No p-values or multiple-comparison-adjusted significance test were calculated.",
             "The source accuracy labels use the original historical OR/equality/containment criterion, not strict AND or the assistant-reviewed false-positive corrections.", "",
             "| Channel | Group-center SNR (dB) | Unicast accuracy (%) | Multicast accuracy (%) | Difference (pp) | Pointwise 95% CI (pp) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['channel']} | {row['group_center_snr_db']:g} | {100*row['unicast_accuracy']:.2f} | "
                     f"{100*row['multicast_accuracy']:.2f} | {row['difference_pp']:+.2f} | [{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] |")
    lines += ["", "## Limitations", "",
              "These intervals quantify between-context sampling uncertainty conditional on this selected subgroup, its task-aware clustering, fixed seed outcomes, simulator assumptions and metric definition. They do not quantify annotation bias, training-selection bias, model mismatch, all possible channel realizations or uncertainty from assistant factual review.",
              "An interval containing zero does not establish equivalence. An interval excluding zero at one point is not a multiple-comparison-adjusted claim across the complete SNR sweep.",
              "No original outcomes, predictions, plots or experiment reports were overwritten.", "",
              "Reproduce: `python -m task_aligned_multicast_500.bootstrap_accuracy_difference`."]
    (OUT / "CONFIDENCE_INTERVAL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    print(summary.to_string(index=False))
    print(OUT / "CONFIDENCE_INTERVAL_REPORT.md")


if __name__ == "__main__":
    main()
