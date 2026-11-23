"""Re-score saved multi-receiver QA outcomes; do not change predictions or plots."""
import ast
import hashlib
import json
from pathlib import Path
import re
import string

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results/and_rule_sensitivity"
RULES = ["historical_accuracy", "strict_and_accuracy", "exceptions_or_and_accuracy"]


def norm(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def rescore(frame, references):
    frame = frame.copy()
    both = frame.f1.ge(.5) & frame.semantic_similarity.ge(.75)
    exceptions = np.asarray([any(norm(ref) == norm(pred) or norm(ref) in norm(pred) for ref in references[str(qid)])
                             for qid, pred in zip(frame.receiver_id, frame.prediction, strict=True)])
    historical = exceptions | frame.f1.ge(.5).to_numpy() | frame.semantic_similarity.ge(.75).to_numpy()
    if not np.array_equal(historical.astype(int), frame.accuracy):
        raise ValueError("Existing accuracy does not match the historical criterion")
    frame["historical_accuracy"] = historical.astype(int)
    frame["strict_and_accuracy"] = both.astype(int)
    frame["exceptions_or_and_accuracy"] = (exceptions | both).astype(int)
    assert frame.strict_and_accuracy.le(frame.historical_accuracy).all()
    assert frame.exceptions_or_and_accuracy.le(frame.historical_accuracy).all()
    return frame


def summarize(frame, by):
    means = frame.groupby(by)[RULES].mean()
    counts = frame.groupby(by)[RULES].sum().rename(columns={name: name.replace("accuracy", "correct_outcomes") for name in RULES})
    sizes = frame.groupby(by).size().rename("outcomes")
    result = means.join(counts).join(sizes).reset_index()
    result["strict_and_change_pp"] = 100*(result.strict_and_accuracy-result.historical_accuracy)
    result["exceptions_or_and_change_pp"] = 100*(result.exceptions_or_and_accuracy-result.historical_accuracy)
    return result


def table(frame, key):
    lines = [f"| {key} | Current accuracy (%) | Strict AND (%) | Change (pp) | Exceptions + AND (%) |",
             "|---|---:|---:|---:|---:|"]
    for r in frame.to_dict("records"):
        label = {"global_alignment": "Semantic alignment", "task_alignment": "Task alignment",
                 "individual_tasks": "Individual task reference"}.get(r[key], r[key])
        lines.append(f"| {label} | {100*r['historical_accuracy']:.2f} | {100*r['strict_and_accuracy']:.2f} | "
                     f"{r['strict_and_change_pp']:+.2f} | {100*r['exceptions_or_and_accuracy']:.2f} |")
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    alignment = ROOT / "paper_multi_receiver_task_alignment/results/angular_misalignment_global_sa_fresh500"
    full = HERE / "results/full"
    subset = HERE / "results/non_singleton_receivers"
    archive = ROOT / "fresh_500_policy_comparison/results_attention_balanced_token_policy/all_strategy_sample_results.csv"
    paths = [alignment/"receiver_results.csv", archive, full/"receiver_results.csv", full/"clean_receiver_results.csv",
             subset/"included_receivers.csv"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    reference_frame = pd.read_csv(archive, usecols=["dataset_index", "true_answers"], keep_default_na=False)
    reference_frame = reference_frame.drop_duplicates("dataset_index")
    references = {str(int(r.dataset_index)): json.loads(r.true_answers) for r in reference_frame.itertuples()}
    original = pd.read_csv(alignment/"receiver_results.csv", keep_default_na=False,
                           dtype={"context_id": str, "receiver_id": str})
    original = rescore(original, references)
    question_summary = summarize(original, ["strategy", "target_removal_percent"])
    original.to_csv(OUT/"clean_optimisation_rescored_rows.csv", index=False)
    question_summary.to_csv(OUT/"clean_optimisation_question_weighted.csv", index=False)
    per_context = original.groupby(["strategy", "target_removal_percent", "context_id"])[RULES].mean().reset_index()
    context_summary = per_context.groupby(["strategy", "target_removal_percent"])[RULES].mean().reset_index()
    context_summary["strict_and_change_pp"] = 100*(context_summary.strict_and_accuracy-context_summary.historical_accuracy)
    context_summary.to_csv(OUT/"clean_optimisation_context_weighted.csv", index=False)
    expected = pd.read_csv(alignment/"all_questions_strategy_summary.csv").set_index(["strategy", "target_removal_percent"])
    np.testing.assert_allclose(question_summary.set_index(["strategy", "target_removal_percent"]).historical_accuracy.sort_index(),
                               expected.mean_accuracy.sort_index(), rtol=0, atol=1e-14)
    cohort = pd.read_csv(subset/"included_receivers.csv", dtype={"context_id": str, "receiver_id": str})
    channels = pd.read_csv(full/"receiver_results.csv", keep_default_na=False,
                           dtype={"context_id": str, "receiver_id": str})
    # Validate references against the transmitted-experiment records as well as the archive.
    assert all(ast.literal_eval(r.references) == references[r.receiver_id] for r in channels.itertuples())
    channels = channels.merge(cohort[["context_id", "receiver_id"]], on=["context_id", "receiver_id"], validate="many_to_one")
    channels = rescore(channels, references)
    channel_summary = summarize(channels, ["channel", "snr_db", "scenario"])
    channel_summary.to_csv(OUT/"non_singleton_channel_question_weighted.csv", index=False)
    channels[["context_id", "receiver_id", "channel", "snr_db", "channel_seed", "scenario", "f1", "semantic_similarity"]+RULES].to_csv(
        OUT/"non_singleton_channel_rescored_rows.csv", index=False)
    seed_means = channels.groupby(["channel", "snr_db", "scenario", "channel_seed"])[RULES].mean()
    np.testing.assert_allclose(seed_means.groupby(["channel", "snr_db", "scenario"]).mean().sort_index(),
                               channel_summary.set_index(["channel", "snr_db", "scenario"])[RULES].sort_index(), rtol=0, atol=1e-14)
    clean = pd.read_csv(full/"clean_receiver_results.csv", keep_default_na=False,
                        dtype={"context_id": str, "receiver_id": str})
    clean = clean.merge(cohort[["context_id", "receiver_id"]], on=["context_id", "receiver_id"], validate="many_to_one")
    clean = rescore(clean, references)
    clean_summary = summarize(clean, ["scenario"])
    clean_summary.to_csv(OUT/"non_singleton_clean_question_weighted.csv", index=False)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    (OUT/"source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    at40 = question_summary[question_summary.target_removal_percent.eq(40)]
    ctx40 = context_summary[context_summary.target_removal_percent.eq(40)]
    average = original.groupby("strategy")[RULES].mean().reset_index()
    average["strict_and_change_pp"] = 100*(average.strict_and_accuracy-average.historical_accuracy)
    average.to_csv(OUT/"clean_optimisation_average_over_levels.csv", index=False)
    names = {"global_alignment": "Semantic alignment", "task_alignment": "Task alignment", "individual_reference": "Individual task reference"}
    at40 = at40.copy()
    at40["strategy"] = at40.strategy.map(lambda s: names.get(s, s))
    report = "\n".join([
        "# AND-rule sensitivity of multi-receiver QA accuracy", "",
        "This is post-hoc metric re-scoring of saved predictions only. No source predictions, plots, model parameters, clusters or channel outcomes were changed.", "",
        "## Exact definitions", "",
        "- Current: normalized equality OR normalized reference containment OR F1 >= 0.5 OR semantic similarity >= 0.75.",
        "- Strict AND: F1 >= 0.5 AND semantic similarity >= 0.75, without other exceptions.",
        "- Exceptions + AND: normalized equality OR normalized reference containment OR (F1 >= 0.5 AND semantic similarity >= 0.75).",
        "The last definition isolates replacing OR by AND while preserving the earlier exact/containment branches. Strict AND is reported separately to avoid ambiguity.",
        "F1 and similarity are the saved best-reference scores. With multiple references their maxima need not come from the same reference; this analysis does not impose a same-reference condition.", "",
        "## A. Original clean multi-receiver optimisation", "",
        "Source: `paper_multi_receiver_task_alignment/results/angular_misalignment_global_sa_fresh500/receiver_results.csv`.",
        "500 questions / 36 contexts; removal targets 5--50%. These are the original clean optimisation curves, not the new fixed-token-ID channel experiment.",
        "Means below give equal weight to questions. Context-weighted results are supplied separately because the original experiment also exported that estimand.", "",
        "### At 40% target removal: question-weighted", "", table(at40, "strategy"), "",
        "### At 40% target removal: context-weighted", "", table(ctx40, "strategy"), "",
        "### Average over all ten removal targets: question-weighted", "", table(average, "strategy"), "",
        "All levels are supplied in `clean_optimisation_question_weighted.csv` and `clean_optimisation_context_weighted.csv`.", "",
        "## B. Latest non-singleton multicast versus unicast channel experiment", "",
        "416 questions / 30 contexts / 109 clusters. Both curves exclude the same 84 singleton receivers. Nominal 40% removal; QPSK; seeds 42, 123, 2024.",
        "For each SNR/strategy/channel there are 1,248 outcomes. Question-weighted then seed-weighted means are unchanged except for the correctness rule.", "",
        "### Clean compressed-message baseline", "", table(clean_summary, "scenario"), "",
    ])
    for channel in ["awgn", "fading"]:
        part = channel_summary[channel_summary.channel.eq(channel)].copy()
        part["condition"] = part.apply(lambda r: f"{r.snr_db:g} dB / {r.scenario}", axis=1)
        report += "\n### " + ("AWGN" if channel == "awgn" else "TDL multipath fading") + "\n\n" + table(part, "condition") + "\n"
    report += "\n## Interpretation\n\nThe decrease quantifies sensitivity to a stricter metric definition, not new channel errors or a change in model outputs. It does not establish which disputed predictions are factually right. Neither rule replaces a manual correctness audit. No new confidence interval or significance claim is made. F1, semantic-similarity means, BER, coding and transmission budgets are unchanged.\n"
    path = OUT/"AND_RULE_SENSITIVITY_REPORT.md"
    path.write_text(report, encoding="utf-8")
    print("Original clean optimisation at 40% (question weighted):")
    print(at40[["strategy"]+RULES+["strict_and_change_pp"]].to_string(index=False))
    print("Latest non-singleton clean baseline:")
    print(clean_summary[["scenario"]+RULES+["strict_and_change_pp"]].to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
