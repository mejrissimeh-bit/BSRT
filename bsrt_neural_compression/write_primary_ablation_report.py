"""Report only primary ablations of the 40%-trained full BSRT reference."""
import json
from pathlib import Path

import pandas as pd

from .plot import markdown_table


def main():
    root = Path(__file__).resolve().parent / "results" / "controlled_ablation"
    ids = ["A0", "A1", "A2", "A3", "A4"]
    metrics = pd.read_csv(root / "metrics.csv")
    metrics = metrics[metrics.id.isin(ids)].copy()
    if len(metrics) != 50 or not metrics.questions.eq(500).all():
        raise ValueError("Expected ten complete 500-question levels for each primary model")
    tests = pd.read_csv(root / "paired_f1_tests_40.csv")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    params = pd.DataFrame([{k: row[k] for k in ["id", "allocated_parameters", "trainable_parameters"]}
                           for row in protocol if row["id"] in ids])
    selected = []
    for name in ids:
        summary = json.loads((root / name / "training_summary.json").read_text(encoding="utf-8"))
        selected.append({"Model": name, "Selected stage": summary["best_stage"],
                         "Epoch": summary["best_epoch"], "Validation F1": summary["best_validation_f1"]})
    lines = [
        "# Primary ablations of the full-reference BSRT model",
        "",
        "## Scope",
        "",
        "This report contains ONLY A0-A4: the full-reference model trained at 40% removal "
        "and its matched primary ablations. It excludes all supplementary multi-budget "
        "and historical distillation experiments. A means ablation; these identifiers are "
        "experimental labels, not different baseline methods. A0 is the highest-validation-F1 "
        "primary variant, not the winner on every final-test metric.",
        "",
        "## Controlled setup",
        "",
        "The compressor training partition contains 256 contexts and 1,980 questions from "
        "SQuAD 2.0 training data. Validation contains 12 separate contexts and 88 questions. "
        "Splits are article-grouped, with the final evaluation question IDs and context hashes "
        "excluded. All primary models train at 40% removal, with eight optimization epochs, "
        "AdamW learning rate 0.0005, weight decay 0.01, gradient clipping 1.0, one context-budget "
        "example per update, and seed 20261103. Checkpoints are selected by downstream "
        "validation context-mean QA F1, not final-test performance.",
        "",
        "The frozen context encoder and downstream QA model are "
        "deepset/bert-base-cased-squad2. Full BSRT uses 768-dimensional context-only hidden "
        "states projected to width 96, learned positions, eight structural token features, "
        "a learned budget embedding, and a two-layer, four-head retention Transformer with "
        "dropout 0.1. It produces token retention logits. Hard selection preserves whole "
        "WordPiece groups at the nearest feasible requested budget and restores selected "
        "source spans in their original order. Encoder-inserted CLS/SEP are excluded.",
        "",
        "A0 reuses the existing full-reference checkpoint trained from scratch. A1-A4 "
        "also start from fresh retention-network initialization, never A0's trained weights. "
        "Compatible shared modules use the same seeded initial values. The frozen BERT "
        "encoder, data partitions, and final cohort remain identical. Separate output folders "
        "preserve configurations, checkpoints, and predictions.",
        "",
        "## Variants",
        "",
        "| Model | Exact modification |",
        "| --- | --- |",
        "| A0 | Full-reference answer-span-supervised BSRT. |",
        "| A1 | Replace the trainable retention Transformer with the existing token-wise MLP; the frozen contextual BERT encoder remains. |",
        "| A2 | Remove the learned budget-embedding contribution; hard budget-constrained selection still enforces the requested removal. |",
        "| A3 | Remove the entire structural-feature projection contribution, including its bias; contextual states, learned positions, and budget embedding remain. |",
        "| A4 | Remove answer-span supervision and train a fresh full architecture using QA-reward-guided mask search and reward-weighted target distillation, with no answer-span loss or answer-span warm start. |",
        "",
        "A0-A3 use labels marking the union of annotated training answer spans per context, "
        "positive-weighted binary cross-entropy, and a budget penalty. A4 instead searches "
        "masks using mean training-question QA F1 and learns the resulting soft token targets "
        "for eight epochs. It uses the training questions and reference answers to evaluate "
        "reward, but never answer-span token labels. Its initial random checkpoint is not "
        "eligible to win selection. A4 is therefore a trained alternative-objective comparison, "
        "not an untrained control or a pure equal-compute component deletion: search adds compute.",
        "",
        "### Parameter counts",
        "",
        markdown_table(params),
        "",
        "Counts cover the retention network only, excluding frozen BERT. Disabled modules "
        "in A2/A3 remain allocated but are frozen and omitted from forward scoring. A1 has "
        "fewer parameters, so its comparison changes architecture and model capacity.",
        "",
        "### Validation-selected checkpoints",
        "",
        markdown_table(pd.DataFrame(selected)),
        "",
        "## Evaluation and metrics",
        "",
        "Every model is evaluated unchanged on the same historical 500 answerable SQuAD "
        "v1.1 validation questions from 36 contexts, at 5%, 10%, ..., 50% removal. The primary "
        "matched-budget comparison is 40%; other levels assess transfer beyond the single "
        "training budget, not multi-budget training. Means weight questions equally. Each "
        "context-budget compression is reused across that context's questions. Deployment "
        "uses context and removal fraction only, without evaluation questions or answers.",
        "",
        "F1 measures normalized answer-token overlap. Semantic similarity is answer embedding "
        "cosine similarity using all-MiniLM-L6-v2. Binary accuracy follows the existing rule: "
        "normalized exact equality, reference substring in the prediction, F1 >= 0.5, OR "
        "semantic similarity >= 0.75. It is not benchmark Exact Match or human-audited "
        "correctness. All variants retain these definitions.",
        "",
        "## Primary results at 40% removal",
        "",
        markdown_table(metrics[metrics.removal_percent.eq(40)][
            ["id", "accuracy", "f1", "semantic_similarity", "actual_token_reduction"]]),
        "",
        "A0 reaches 81.2% accuracy and 0.7498 F1. Replacing the retention Transformer "
        "reduces mean F1 by 0.0111, and removing budget conditioning reduces it by 0.0121. "
        "Both variants nevertheless have slightly higher observed accuracy (81.6%). Removing "
        "structural features increases observed F1 by 0.0085 and accuracy by 1.8 percentage "
        "points. Thus the full architecture is not uniformly best on the final cohort.",
        "",
        "A4 reaches 63.6% accuracy and 0.5321 F1: 17.6 percentage points lower accuracy "
        "and 0.2177 lower F1 than A0. This is evidence favoring the tested answer-span "
        "supervision procedure over the tested fresh reward-only procedure, not proof that "
        "all possible reward-based learning methods are inferior.",
        "",
        "## Statistical comparisons at 40%",
        "",
        "The estimand is mean paired question-level F1 difference (A0 minus each variant). "
        "Questions are paired by identifier and grouped by context. Percentile 95% confidence "
        "intervals resample 36 contexts with replacement 10,000 times, retaining all questions "
        "in each context and calculating question-weighted means. Two-sided context sign-flip "
        "tests use 50,000 random sign assignments and the plus-one Monte Carlo p-value "
        "correction. All question differences within a context receive the same sign. "
        "The test assumes independent contexts and sign symmetry under the null; it is not "
        "a paired t-test. Seed: 20261103. Holm correction covers the four planned primary "
        "comparisons at significance level alpha = 0.05. Confidence intervals are pointwise, "
        "not simultaneous familywise intervals.",
        "",
        "Null hypothesis for each comparison: no difference in expected F1 between A0 "
        "and the corresponding variant. The table reports full-precision Monte Carlo "
        "p-values separately to avoid displaying small nonzero values as zero.",
        "",
        "| Comparison | Mean F1 difference | 95% CI | Raw p | Holm p | Significant at 0.05? |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in tests.itertuples(index=False):
        lines.append(f"| {row.comparison} | {row.mean_f1_difference:.6f} | "
                     f"[{row.ci_low:.6f}, {row.ci_high:.6f}] | "
                     f"{row.cluster_signflip_p:.10g} | {row.holm_p:.10g} | "
                     f"{'Yes' if row.holm_p < .05 else 'No'} |")
    lines += ["", "A1-A3: intervals include zero and Holm p-values equal 1.0. There is no "
              "statistically established advantage of A0 over these alternatives. This does "
              "not establish equivalence or prove that their removed components never help. "
              "A4: the interval [0.1607, 0.2697] excludes zero and Holm p is approximately "
              "0.000080, supporting a substantial difference under this protocol.", "",
              "## Complete cross-budget results", "",
              "Values are on a 0-1 scale. Each cell is the mean of 500 question outcomes. "
              "These transfer-level tables are descriptive; no additional hypothesis tests "
              "are claimed at these budgets."]
    for metric, title in [("accuracy", "Binary accuracy"), ("f1", "Answer F1"),
                          ("semantic_similarity", "Answer semantic similarity")]:
        table = metrics.pivot(index="removal_percent", columns="id", values=metric)[ids].reset_index()
        lines += ["", f"### {title}", "", markdown_table(table)]
    lines += ["", "## Conclusions and limits", "",
              "- Answer-span-supervised A0 clearly outperforms the tested reward-only A4 "
              "at the common 40% training budget. A4 changes the objective and uses additional "
              "search compute, so describe it as an alternative-supervision experiment.",
              "- The results do not show a significant F1 benefit for the retention "
              "Transformer, explicit budget embedding, or structural features at 40%.",
              "- A3 is slightly better on the observed test cohort, but its validation "
              "score is lower and its F1 advantage is not significant. A0 remains the "
              "validation-selected primary reference; do not call it the universally best model.",
              "- Because all models train at one budget, A2 does not establish whether "
              "conditioning helps when learning several budgets jointly.",
              "- Only one initialization seed is used. Context-resampling intervals do "
              "not quantify training-seed uncertainty. A1 also changes parameter count.",
              "- The historical 500 questions were repeatedly inspected in previous thesis "
              "development. They are held out from this training process, but are not a "
              "pristine untouched discovery test. There are only 36 evaluation contexts.",
              "- Permissive binary correctness thresholds can accept factually incorrect "
              "answers. F1 and semantic similarity provide additional, distinct evidence.", "",
              "## Source records and plots", "",
              "Paths below are relative to Codes/bsrt_neural_compression/results/controlled_ablation/.", "",
              "- metrics.csv: this report filters ONLY A0-A4 (50 complete model-level records).",
              "- paired_f1_tests_40.csv: four primary clustered comparisons.",
              "- protocol.json: per-variant parameter counts and matched settings.",
              "- A0/ through A4/: config.json, training_summary.json, checkpoints/, "
              "evaluation/final_500/bsrt_sample_results.csv, and comparison_summary.csv.",
              "- primary_accuracy.pdf, primary_f1.pdf, primary_semantic_similarity.pdf: "
              "ONLY primary models; SVG and PNG copies are also available.",
              "- Implementation: ../../controlled_ablation.py, ../../model.py, "
              "../../training.py, and ../../evaluate.py.", "",
              "This report is generated from completed records. It does not modify the "
              "thesis, recalculate experiments, or include supplementary-model results.", ""]
    destination = root / "PRIMARY_MODEL_ABLATIONS_REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
