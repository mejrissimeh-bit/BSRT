"""Build a source-linked report from completed full-reference results only."""
from pathlib import Path

import pandas as pd

from .plot import markdown_table


def main():
    root = Path(__file__).resolve().parent
    output = root / "results" / "controlled_ablation" / "A0"
    evaluation = output / "evaluation" / "final_500"
    frame = pd.read_csv(evaluation / "comparison_summary.csv")
    paired = pd.read_csv(evaluation / "paired_f1_differences.csv")
    methods = ["random", "attention", "l2", "learned_token_policy", "bsrt", "existing_question_oracle"]
    names = {"random": "Random", "attention": "Attention", "l2": "L2",
             "learned_token_policy": "Previous token policy", "bsrt": "Full BSRT",
             "existing_question_oracle": "Existing question-aware oracle"}
    lines = [
        "# Full-reference BSRT: final 500-question comparison",
        "",
        "## Scope and model selection",
        "",
        "This report describes one fixed, full-reference BSRT checkpoint (A0). "
        "A0 is an experiment identifier, not part of the model name. No ablation variants, "
        "distilled variants, or per-removal-level checkpoint switching are included. "
        "Among the primary A0-A4 variants, A0 has the highest validation F1. "
        "It is not the highest observed final-test variant at every level; model selection "
        "here follows validation rather than choosing the winner on the final 500 questions.",
        "",
        "## Training and deployable inference",
        "",
        "Training uses 256 contexts containing 1,980 questions from SQuAD 2.0 training data. "
        "Compressor validation uses 12 separate contexts with 88 questions. Splitting is "
        "article-grouped; final evaluation question IDs and context hashes are excluded. "
        "These splits establish separation for the compressor, not an unseen-data guarantee "
        "for the pretrained QA teacher. The seed is 20261103; this is one training-seed experiment.",
        "",
        "The frozen encoder and downstream QA model are deepset/bert-base-cased-squad2. "
        "The encoder receives context only and supplies 768-dimensional contextual hidden states. "
        "The trainable retention network projects to width 96 and combines structural features, "
        "position embeddings, and a learned budget embedding. A two-layer, four-head Transformer "
        "with dropout 0.1 produces a scalar retention logit per token. A0 has 320,353 trainable "
        "retention-network parameters; the frozen BERT parameters are additional.",
        "",
        "Training targets mark tokens overlapping the union of annotated answer spans across "
        "the training questions for a context. Weighted binary cross-entropy is combined with "
        "0.1 times the squared discrepancy between mean retention probability and the target "
        "retained fraction. Training uses AdamW, learning rate 0.0005, weight decay 0.01, "
        "gradient clipping at 1.0, and one context-budget example per update. Eight epochs were "
        "run. The selected checkpoint is pretraining epoch 7, with validation context-mean "
        "QA F1 0.7423295012. It was trained from scratch, without warm-starting or guided-search "
        "distillation. Questions and reference answers provide offline training supervision only.",
        "",
        "IMPORTANT: this successful checkpoint was trained and selected at 40% removal only. "
        "The same unchanged checkpoint is evaluated at all ten removal targets below. "
        "Therefore, describe the other budgets as cross-budget evaluation, not multi-budget training.",
        "",
        "Inference takes only context and requested removal fraction. It requires a frozen "
        "BERT context encoding and retention-network inference, but no question, answer, "
        "downstream QA run, or online mask search. Whole WordPiece groups are indivisible. "
        "Budget-constrained selection maximizes summed selected logits at the requested "
        "retained token count, or the nearest feasible count. Selected source spans retain "
        "their original order. Encoder-inserted CLS/SEP are not selection candidates. "
        "The maximum context input is 510 source WordPieces, plus encoder special tokens.",
        "",
        "## Final evaluation and metric definitions",
        "",
        "The comparison uses the same 500 answerable question-context pairs from 36 distinct "
        "contexts in the historical SQuAD v1.1 validation archive. Do not call this final "
        "subset SQuAD 2.0: the training corpus and final historical archive differ. Each method "
        "has 500 records per target, at 5%, 10%, ..., 50% removal. Means weight questions "
        "equally, not contexts equally. A single context-only compression is reused for all "
        "questions associated with that context and budget. Results are not averages over "
        "multiple training or channel seeds; no channel is simulated in this experiment.",
        "",
        "Answer F1 is normalized token-overlap F1 against the reference answers. Semantic "
        "similarity uses all-MiniLM-L6-v2 sentence embeddings and cosine similarity of answers, "
        "not context-text similarity. Binary accuracy follows the existing project rule: "
        "normalized equality, reference-answer substring in the prediction, F1 >= 0.5, OR "
        "semantic similarity >= 0.75. It is not benchmark Exact Match and must not be described "
        "as independently human-verified correctness. These permissive conditions can accept "
        "factually incorrect predictions, particularly numerical substitutions.",
        "",
        "The plots include a common 0% point from the recorded uncompressed answers. This "
        "is a shared reference anchor, not a separate original-text curve. The tables below "
        "show the actual compressed evaluation levels only.",
        "",
        "## Quantitative comparison",
        "",
        "All values are proportions on a 0-1 scale. Rounded to four decimal places; source "
        "CSVs retain full precision.",
    ]
    for metric, title in [("accuracy", "Binary accuracy"), ("f1", "Answer F1"),
                          ("semantic_similarity", "Answer semantic similarity")]:
        table = frame.pivot(index="removal_percent", columns="method", values=metric)
        table = table[methods].rename(columns=names).reset_index()
        lines += ["", f"### {title}", "", markdown_table(table)]
    actual = frame[frame.method.eq("bsrt")][["removal_percent", "actual_token_reduction"]].copy()
    actual["actual_token_reduction"] *= 100
    actual.columns = ["Target removal (%)", "Mean measured token removal (%)"]
    lines += ["", "### Requested versus measured reduction", "", markdown_table(actual), "",
              "Measured reduction uses re-tokenized reconstructed text. Integer budgets, "
              "whole-word grouping, and text reconstruction explain deviations from target "
              "percentages. Token removal is not a measured bit-rate reduction.", "",
              "## Interpretation supported by these results", "",
              "BSRT has higher mean answer F1 than Random, Attention, L2, and the previous "
              "token policy at every evaluated compressed level. This does not imply that "
              "it wins on every question, every metric, or every training seed. At low removal, "
              "its binary accuracy is not consistently the highest: for example, the previous "
              "policy reaches 0.912 at 5%, versus 0.900 for BSRT; at 15%, L2 reaches 0.898 "
              "versus 0.870 for BSRT.", "",
              "The clearest advantages occur at aggressive removal. At 40%, BSRT accuracy "
              "is 0.812 versus 0.726 for the previous policy and 0.692 for L2: gains of 8.6 "
              "and 12.0 percentage points. F1 is 0.7498 versus 0.6306 and 0.5799; semantic "
              "similarity is 0.8506 versus 0.7837 and 0.7536. At 50%, BSRT accuracy is 0.772 "
              "versus 0.600 and 0.542, and F1 is 0.6813 versus 0.4785 and 0.4486. These "
              "are descriptive paired-cohort differences, not by themselves significance tests.", "",
              "## Paired uncertainty estimates", "",
              "The following existing intervals use paired question-level F1 differences, "
              "resampling all 36 contexts with replacement 10,000 times, retaining all questions "
              "within each sampled context. Each replicate divides summed differences by its "
              "question count, preserving a question-weighted estimand. The bounds are the "
              "2.5th and 97.5th percentiles; seed 20261103. They account for shared contexts, "
              "but not training-seed variation. These are pointwise 95% percentile bootstrap "
              "intervals, not simultaneous bands or multiplicity-corrected hypothesis tests. "
              "No baseline-comparison p-values are established by this file.", "",
              markdown_table(paired), "",
              "At 40%, the paired F1 gain over the previous policy is 0.1191, with 95% CI "
              "[0.0613, 0.1823]; over L2 it is 0.1699, CI [0.1133, 0.2292]. At 50%, the "
              "corresponding gains are 0.2028 [0.1285, 0.2796] and 0.2328 [0.1614, 0.3041].", "",
              "## Oracle interpretation", "",
              "The plotted oracle is the unchanged question-aware, outcome-dependent reference "
              "over Random, Attention, L2, and the previous learned token policy. It does NOT "
              "include BSRT. Selection maximizes the archived utility, rather than independently "
              "maximizing each plotted metric. Consequently it is an empirical hindsight "
              "reference for that candidate set, not a strict upper bound on arbitrary new "
              "policies or on every metric. It is not deployable. At 40%, its accuracy/F1/semantic "
              "similarity are 0.924/0.8309/0.9243, compared with BSRT 0.812/0.7498/0.8506. "
              "The question-blind inference constraint can contribute to the remaining gap; "
              "the data do not prove that the entire gap is unavoidable or caused only by blindness.", "",
              "## Limitations and writing guidance", "",
              "- The final 500 are excluded from new training and validation but have been "
              "inspected repeatedly throughout thesis development. They are a fixed historical "
              "comparison cohort, not a pristine untouched discovery test.",
              "- There are only 36 distinct evaluation contexts and one training seed. Avoid "
              "claiming universal generalization or seed robustness.",
              "- The answer-span objective teaches offline answer-bearing-token preservation; "
              "the deployed compressor does not know evaluation questions or answers.",
              "- The original auxiliary held-out 12-context/88-question assessment did not "
              "show the same strong F1 improvement over the previous policy (approximately "
              "0.6945 versus 0.6927). It is a separate cohort, not pooled into these 500 results.",
              "- Cached BSRT compression timings in the sample CSV exclude cold BERT encoding. "
              "Do not use them to claim an end-to-end speed advantage over baselines.",
              "- No communication-channel robustness, transmission-cost saving, or error "
              "protection conclusion follows from this text-compression experiment alone.",
              "- For a thesis section, present full BSRT as a context-only, answer-span-supervised "
              "neural retention model; do not attribute these results to guided search, "
              "distillation, or multi-budget training.", "",
              "## Exact sources and figures", "",
              "All paths below are relative to Codes/bsrt_neural_compression/.", "",
              "- Original training configuration: config_span_256_40.json.",
              "- Original checkpoint: results/span_256_40/checkpoints/best.pt (pretrain epoch 7).",
              "- Reused full-reference configuration: results/controlled_ablation/A0/config.json.",
              "- Training selection: results/controlled_ablation/A0/training_summary.json.",
              "- Aggregates: results/controlled_ablation/A0/evaluation/final_500/comparison_summary.csv.",
              "- Paired records: results/controlled_ablation/A0/evaluation/final_500/all_comparison_sample_results.csv.",
              "- BSRT records: results/controlled_ablation/A0/evaluation/final_500/bsrt_sample_results.csv.",
              "- Bootstrap intervals: results/controlled_ablation/A0/evaluation/final_500/paired_f1_differences.csv.",
              "- Provenance: results/controlled_ablation/A0/evaluation/final_500/frozen_checkpoint.json.",
              "- Architecture/selection: model.py. Training targets/loss/selection: training.py.",
              "- Evaluation and clustered bootstrap: evaluate.py. QA and similarity: reward.py.",
              "- Historical correctness rule: ../main.py, is_correct_answer().",
              "- Plot generation: plot.py; plot data: results/controlled_ablation/A0/plots/final_500_plot_data.csv.",
              "- Figures: results/controlled_ablation/A0/plots/final_500_accuracy.pdf, "
              "final_500_f1.pdf, final_500_semantic_similarity.pdf (also SVG and PNG).",
              "- This report does not insert figures into LaTeX or create new thesis labels.", ""]
    destination = output / "FULL_REFERENCE_FINAL_500_REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
