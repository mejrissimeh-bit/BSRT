"""Inspect low-F1 outcomes accepted by the historical correctness criterion."""
import ast
from pathlib import Path

import pandas as pd

from .and_rule_sensitivity import norm


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/full"
OUT = HERE / "results/and_rule_sensitivity"


def inspect(frame, references):
    frame = frame.copy()
    frame["references_list"] = frame.receiver_id.map(references)
    frame["exact_match_branch"] = [any(norm(ref) == norm(pred) for ref in refs)
                                   for pred, refs in zip(frame.prediction, frame.references_list, strict=True)]
    frame["containment_branch"] = [any(norm(ref) in norm(pred) for ref in refs)
                                   for pred, refs in zip(frame.prediction, frame.references_list, strict=True)]
    frame["similarity_branch"] = frame.semantic_similarity.ge(.75)
    frame["low_f1_accepted"] = frame.accuracy.eq(1) & frame.f1.lt(.5)
    assert ((frame.exact_match_branch | frame.containment_branch | frame.similarity_branch | frame.f1.ge(.5)).astype(int) == frame.accuracy).all()
    frame["low_f1_similarity_only"] = frame.low_f1_accepted & frame.similarity_branch & ~frame.containment_branch
    frame["low_f1_containment_only"] = frame.low_f1_accepted & frame.containment_branch & ~frame.similarity_branch
    frame["low_f1_both"] = frame.low_f1_accepted & frame.containment_branch & frame.similarity_branch
    frame["zero_f1_accepted"] = frame.low_f1_accepted & frame.f1.eq(0)
    return frame


def summarize(frame, keys):
    counts = frame.groupby(keys)[["accuracy", "low_f1_accepted", "zero_f1_accepted", "low_f1_similarity_only",
                                  "low_f1_containment_only", "low_f1_both"]].sum()
    counts = counts.rename(columns={"accuracy": "labelled_correct_outcomes"})
    counts["total_outcomes"] = frame.groupby(keys).size()
    counts["low_f1_percent_of_labelled_correct"] = 100*counts.low_f1_accepted/counts.labelled_correct_outcomes
    counts["low_f1_percent_of_all_outcomes"] = 100*counts.low_f1_accepted/counts.total_outcomes
    assert (counts.low_f1_accepted == counts.low_f1_similarity_only+counts.low_f1_containment_only+counts.low_f1_both).all()
    return counts.reset_index()


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    cohort = pd.read_csv(HERE / "results/non_singleton_receivers/included_receivers.csv",
                         dtype={"context_id": str, "receiver_id": str})
    raw = pd.read_csv(SOURCE / "receiver_results.csv", keep_default_na=False,
                      dtype={"context_id": str, "receiver_id": str})
    references = {r.receiver_id: ast.literal_eval(r.references) for r in raw.itertuples()}
    records = raw.merge(cohort[["context_id", "receiver_id"]], on=["context_id", "receiver_id"], validate="many_to_one")
    records = inspect(records, references)
    summary = summarize(records, ["channel", "snr_db", "scenario"])
    summary.to_csv(OUT / "low_f1_correct_channel_counts.csv", index=False)
    low = records[records.low_f1_accepted]
    low.to_csv(OUT / "low_f1_correct_channel_records.csv", index=False)
    clean = pd.read_csv(SOURCE / "clean_receiver_results.csv", keep_default_na=False,
                        dtype={"context_id": str, "receiver_id": str})
    clean = clean.merge(cohort[["context_id", "receiver_id"]], on=["context_id", "receiver_id"], validate="many_to_one")
    clean = inspect(clean, references)
    clean_counts = summarize(clean, ["scenario"])
    clean_counts.to_csv(OUT / "low_f1_correct_clean_counts.csv", index=False)
    clean[clean.low_f1_accepted].to_csv(OUT / "low_f1_correct_clean_records.csv", index=False)
    print("Multicast counts (three seeds pooled; 1,248 outcomes per point):")
    print(summary[summary.scenario.eq("multicast")].to_string(index=False))
    print("Clean baseline: 416 unique questions per strategy:")
    print(clean_counts.to_string(index=False))
    examples = clean[clean.scenario.eq("multicast") & clean.low_f1_accepted].copy()
    questions = raw[["receiver_id", "question"]].drop_duplicates("receiver_id")
    examples = examples.merge(questions, on="receiver_id", validate="one_to_one")
    examples[["receiver_id", "question", "references_list", "prediction", "f1", "semantic_similarity",
              "containment_branch", "similarity_branch"]].to_csv(OUT / "low_f1_correct_clean_multicast_examples.csv", index=False)
    print("Clean multicast examples:")
    print(examples[["receiver_id", "question", "references_list", "prediction", "f1", "semantic_similarity",
                    "containment_branch", "similarity_branch"]].to_string(index=False))
    lines = ["# Low-F1 outcomes labelled correct in the latest multi-receiver channel experiment", "",
             "Scope: the paired 416-question, 30-context non-singleton cohort. No predictions or metrics were changed.",
             "Correct here means labelled correct by the implemented historical criterion, not independently verified factual correctness.", "",
             "## Clean compressed-message reference", "", markdown_table(clean_counts), "",
             "## Channel outcomes: multicast", "", markdown_table(summary[summary.scenario.eq("multicast")]), "",
             "Each channel/SNR row contains 416 questions x 3 seeds = 1,248 outcomes, not 1,248 distinct questions.",
             "Clean rows count each of the 416 questions once. Repeated clean receptions at different SNRs/seeds must not be counted as distinct QA examples.", "",
             "## Interpretation", "",
             "Low-F1 accepted outcomes can pass similarity >=0.75, normalized substring containment, or both. Low F1 alone does not prove an answer is wrong.",
             "Containment is a character-substring test, not a token-boundary-aware match. It can accept incorrect partial/numeric matches; similarity can also accept factually different answers.",
             "Concrete recorded cases: receiver 1015 has reference `711,988`, prediction `711, 988`, F1=0 and similarity=1: a formatting mismatch, not a changed number. Receiver 491 has reference `Monday`, prediction `Tuesday`, F1=0 and similarity about 0.87422: a false positive relative to the reference. Receiver 292 predicts `two` for reference `three`, F1=0 and similarity about 0.80616. Receiver 31 predicts `12` for references `8`/`eight`, F1=0 and similarity about 0.75970. Receiver 443 predicts `using Arabic numerals` for reference `L`, F1=0 and similarity about 0.03480, accepted through character containment of `l` in `numerals`.",
             "See `low_f1_correct_clean_multicast_examples.csv` for all affected clean multicast questions and exact recorded answers/scores.",
             "No claim is made that every low-F1 answer is factually wrong or right. That requires answer-by-answer factual review."]
    path = OUT / "LOW_F1_CORRECT_AUDIT.md"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
