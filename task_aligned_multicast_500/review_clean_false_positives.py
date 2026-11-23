"""Record an assistant reference/context review; never alter original accuracy labels."""
import ast
import hashlib
from pathlib import Path

import pandas as pd

from .and_rule_sensitivity import norm


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results/and_rule_sensitivity"

# These decisions concern the saved clean multicast prediction for each question,
# not every prediction of that question at arbitrary SNR/seed settings.
CLEAR_ERRORS = {
    "84": "Wrong event: XLVIII instead of XLVII.",
    "101": "Wrong event: XLVIII instead of XLVII.",
    "186": "Insufficient answer: 'Super Bowl' omits the requested edition XXXVIII.",
    "248": "Wrong player: original passage assigns tackle leadership to Luke Kuechly, not Kawann Short. One reference says Short, but contradicts the passage.",
    "443": "Wrong answer: 'using Arabic numerals' does not give the Roman numeral L. Character-substring matching accepts l inside numerals.",
    "491": "Wrong day: Tuesday instead of Monday.",
    "292": "Wrong count: two instead of three.",
    "31": "Wrong count: 12 instead of 8/eight.",
}
OVERINCLUSIVE = {
    "87": "Adds Bruno Mars although the requested XLVII performer is Beyonce.",
    "88": "Adds Beyonce although the requested XLVIII performer is Bruno Mars.",
    "92": "Adds Beyonce although the question specifically asks for the male guest singer Bruno Mars.",
    "105": "Adds Beyonce although the passage identifies Bruno Mars as XLVIII headliner. References also conflict by listing Coldplay.",
}
ACCEPTABLE = {
    "423": "Contains the correct practice location Stanford University followed by an extraneous compressed fragment; retained as a location answer.",
    "1015": "Same population 711,988; spacing only.",
    "1016": "Same percentage 56.2%; spacing only. One reference is inconsistent.",
    "402": "Same score 34-19; dash spacing only.",
    "407": "Same score 34-19; dash spacing only.",
    "410": "Road white gives the jersey colour; no contradictory colour.",
    "119": "Same venue and area; apostrophe/spacing differences.",
    "121": "Same Levi's Stadium; apostrophe/spacing only.",
    "122": "Same Levi's Stadium; apostrophe/spacing only.",
    "123": "Same Mercedes-Benz Superdome; hyphen spacing only.",
    "130": "Same Levi's Stadium; apostrophe/spacing only.",
    "439": "Longer phrase explicitly supplies Super Bowl L as the Roman-numeral alternative.",
    "466": "Moscone Center with its San Francisco location; correct elaboration.",
    "499": "Monday evening includes the correct requested day.",
    "222": "Same rating 99.4; spacing only.",
    "228": "Same 3,837 yards; spacing only.",
    "232": "Same 3,837 yards; spacing only.",
    "234": "Same rating 99.4; spacing only.",
    "2": "Same Santa Clara, California location with broader Bay Area context.",
    "13": "Same Santa Clara city with broader Bay Area context.",
    "17": "Same Santa Clara city with broader Bay Area context.",
    "19": "Same score 24-10; dash spacing only.",
    "156": "Owners is contextually the NFL owners; no conflicting entity.",
    "158": "Same $1.2 billion; spacing only.",
    "162": "Same $1.2 billion; spacing only.",
    "167": "Same $1.2 billion; spacing only.",
    "176": "Same $1.2 billion; spacing only.",
    "315": "Same score 49-15; dash spacing only.",
    "74": "Von Miller with role/team description; correct entity.",
    "259": "Correct plantar-fasciitis injury with additional passage-supported detail.",
    "262": "Correct plantar-fasciitis injury with heel detail.",
    "275": "Same rating 67.9; spacing only.",
    "276": "Same 2,249 yards; spacing only.",
    "280": "Same rating 67.9; spacing only.",
    "283": "Five touchdowns is present; 720 yards is additional passage-supported information.",
    "284": "Same rating 67.9; spacing only.",
    "290": "Same 5.5 sacks, expressed as 5-and-a-half plus the unit.",
}
AMBIGUOUS = {"277": "Malformed question and conflicting references nine/17/rating. Prediction 17 interceptions agrees with the passage and the most direct reading; not counted as an established false positive."}


def main():
    raw_path = HERE / "results/full/clean_receiver_results.csv"
    source_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw = pd.read_csv(raw_path, keep_default_na=False, dtype={"context_id": str, "receiver_id": str})
    cohort = pd.read_csv(HERE / "results/non_singleton_receivers/included_receivers.csv", dtype={"context_id": str, "receiver_id": str})
    frame = raw.merge(cohort[["context_id", "receiver_id"]], on=["context_id", "receiver_id"], validate="many_to_one")
    frame = frame[frame.scenario.eq("multicast")].copy()
    archive = pd.read_csv(ROOT / "fresh_500_policy_comparison/results_attention_balanced_token_policy/all_strategy_sample_results.csv",
                          usecols=["dataset_index", "question", "true_answers", "original_context"], keep_default_na=False)
    archive = archive.drop_duplicates("dataset_index")
    archive["receiver_id"] = archive.dataset_index.astype(str)
    frame = frame.merge(archive.drop(columns="dataset_index"), on="receiver_id", validate="one_to_one")
    frame["references"] = frame.true_answers.map(ast.literal_eval)
    decisions = {}
    for category, mapping in [("clear_wrong_or_insufficient", CLEAR_ERRORS), ("overinclusive_wrong_entity", OVERINCLUSIVE),
                              ("reviewed_acceptable", ACCEPTABLE), ("ambiguous_not_rejected", AMBIGUOUS)]:
        for qid, reason in mapping.items():
            assert qid not in decisions
            decisions[qid] = (category, reason)
    nonexact = frame[frame.accuracy.eq(1) & ~frame.apply(lambda r: any(norm(ref) == norm(r.prediction) for ref in r.references), axis=1)]
    assert set(nonexact.receiver_id) == set(decisions) and len(nonexact) == 50
    assert len(frame) == 416 and frame.accuracy.sum() == 376
    frame["review_category"] = frame.apply(lambda r: "originally_incorrect" if not r.accuracy else
        decisions[r.receiver_id][0] if r.receiver_id in decisions else "normalized_reference_match", axis=1)
    frame["review_reason"] = frame.receiver_id.map(lambda qid: decisions.get(qid, ("", "Matches a normalized reference; not independently fact-checked beyond the benchmark."))[1])
    frame["accuracy_after_clear_rejections"] = (frame.accuracy.eq(1) & ~frame.receiver_id.isin(CLEAR_ERRORS)).astype(int)
    rejected_strict = set(CLEAR_ERRORS) | set(OVERINCLUSIVE)
    frame["accuracy_after_clear_and_overinclusive_rejections"] = (frame.accuracy.eq(1) & ~frame.receiver_id.isin(rejected_strict)).astype(int)
    assert len(CLEAR_ERRORS) == 8 and len(OVERINCLUSIVE) == 4 and len(ACCEPTABLE) == 37
    frame.to_csv(OUT / "clean_multicast_false_positive_review.csv", index=False)
    rows = []
    for name, rejected in [("Current implemented criterion", set()), ("Reject 8 clear wrong/insufficient answers", set(CLEAR_ERRORS)),
                           ("Also reject 4 overinclusive wrong-entity answers", rejected_strict),
                           ("Additionally reject the ambiguous question (sensitivity only)", rejected_strict | set(AMBIGUOUS))]:
        correct = int(frame.accuracy.sum())-len(rejected)
        rows.append({"scenario": name, "correct_questions": correct, "total_questions": len(frame),
                     "accuracy_percent": 100*correct/len(frame), "drop_percentage_points": 100*len(rejected)/len(frame)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "clean_multicast_false_positive_impact.csv", index=False)
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == source_hash
    text = "\n".join([
        "# False-positive review: latest clean multicast cohort", "",
        "Scope: 416 questions from 30 contexts in non-singleton clusters; nominal 40% token removal. The current clean multicast accuracy is 376/416 = 90.384615%.",
        "The same clean answers occur at every seed from 5 dB AWGN and 7 dB fading. This review does not adjudicate different corrupted predictions at low SNR.", "",
        "## Review method", "",
        "326 accepted answers match a normalized reference; 50 accepted non-exact answers were individually inspected against the question and references, with original passages checked for conflicting identities/events and malformed questions. Sixteen of the 326 normalized matches differ from all literal lowercased references by formatting/articles; these were inspected too.",
        "This is an assistant-assisted reference/context review, not an independent blinded human annotation study. The 326 normalized matches are treated as reference-consistent; that does not prove every underlying benchmark annotation is factually correct.", "",
        "The 50 non-exact decisions are: 37 acceptable answers/formatting variants, 8 clearly wrong or insufficient answers, 4 answers with an incorrect extra entity, and 1 malformed/reference-conflicting question not counted as established false positive.", "",
        "## Numerical impact", "", summary.to_csv(index=False).strip(), "",
        "A strict answer-level judgement rejects 12 outcomes, giving 364/416 = 87.5%, a 2.884615 percentage-point drop. Restricting rejection to the 8 clear wrong/insufficient answers gives 368/416 = 88.461538%, a 1.923077-point drop.",
        "The additional 13th rejection is only a sensitivity scenario for an ambiguous item, not a claim that its predicted 17 interceptions is wrong.", "",
        "## Why AND does not remove all false positives", "",
        "Questions 84 and 101 predict Super Bowl XLVIII instead of XLVII but have F1 about 0.6667 and similarity about 0.94277. Question 186 predicts only Super Bowl instead of Super Bowl XXXVIII, with F1=0.8 and similarity about 0.84185. All three still pass strict AND.",
        "Conversely, some truly equivalent numeric/name answers have low F1 due to punctuation/spacing and would be rejected by strict AND.", "",
        "## Boundaries", "",
        "These corrections are review scenarios, not replacement experimental results. No original CSV, plot, metric or report was modified. The source, clustering, transport, F1, similarity and BER are unchanged.",
        "Do not present 87.5% as a formally established all-false-positives-removed accuracy across every SNR. Low-SNR predictions need their own adjudication; exact-reference outputs can also inherit dataset annotation issues.",
        "See clean_multicast_false_positive_review.csv for every question, reference, original passage, prediction, score and review decision; clean_multicast_false_positive_impact.csv contains scenario arithmetic.",
        f"Source clean-results SHA-256: {source_hash}",
    ])
    path = OUT / "CLEAN_MULTICAST_FALSE_POSITIVE_REVIEW.md"
    path.write_text(text+"\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
