# Primary ablations of the full-reference BSRT model

## Scope

This report contains ONLY A0-A4: the full-reference model trained at 40% removal and its matched primary ablations. It excludes all supplementary multi-budget and historical distillation experiments. A means ablation; these identifiers are experimental labels, not different baseline methods. A0 is the highest-validation-F1 primary variant, not the winner on every final-test metric.

## Controlled setup

The compressor training partition contains 256 contexts and 1,980 questions from SQuAD 2.0 training data. Validation contains 12 separate contexts and 88 questions. Splits are article-grouped, with the final evaluation question IDs and context hashes excluded. All primary models train at 40% removal, with eight optimization epochs, AdamW learning rate 0.0005, weight decay 0.01, gradient clipping 1.0, one context-budget example per update, and seed 20261103. Checkpoints are selected by downstream validation context-mean QA F1, not final-test performance.

The frozen context encoder and downstream QA model are deepset/bert-base-cased-squad2. Full BSRT uses 768-dimensional context-only hidden states projected to width 96, learned positions, eight structural token features, a learned budget embedding, and a two-layer, four-head retention Transformer with dropout 0.1. It produces token retention logits. Hard selection preserves whole WordPiece groups at the nearest feasible requested budget and restores selected source spans in their original order. Encoder-inserted CLS/SEP are excluded.

A0 reuses the existing full-reference checkpoint trained from scratch. A1-A4 also start from fresh retention-network initialization, never A0's trained weights. Compatible shared modules use the same seeded initial values. The frozen BERT encoder, data partitions, and final cohort remain identical. Separate output folders preserve configurations, checkpoints, and predictions.

## Variants

| Model | Exact modification |
| --- | --- |
| A0 | Full-reference answer-span-supervised BSRT. |
| A1 | Replace the trainable retention Transformer with the existing token-wise MLP; the frozen contextual BERT encoder remains. |
| A2 | Remove the learned budget-embedding contribution; hard budget-constrained selection still enforces the requested removal. |
| A3 | Remove the entire structural-feature projection contribution, including its bias; contextual states, learned positions, and budget embedding remain. |
| A4 | Remove answer-span supervision and train a fresh full architecture using QA-reward-guided mask search and reward-weighted target distillation, with no answer-span loss or answer-span warm start. |

A0-A3 use labels marking the union of annotated training answer spans per context, positive-weighted binary cross-entropy, and a budget penalty. A4 instead searches masks using mean training-question QA F1 and learns the resulting soft token targets for eight epochs. It uses the training questions and reference answers to evaluate reward, but never answer-span token labels. Its initial random checkpoint is not eligible to win selection. A4 is therefore a trained alternative-objective comparison, not an untrained control or a pure equal-compute component deletion: search adds compute.

### Parameter counts

| id | allocated_parameters | trainable_parameters |
| --- | --- | --- |
| A0 | 320353 | 320353 |
| A1 | 189409 | 189409 |
| A2 | 320353 | 310753 |
| A3 | 320353 | 319489 |
| A4 | 320353 | 320353 |

Counts cover the retention network only, excluding frozen BERT. Disabled modules in A2/A3 remain allocated but are frozen and omitted from forward scoring. A1 has fewer parameters, so its comparison changes architecture and model capacity.

### Validation-selected checkpoints

| Model | Selected stage | Epoch | Validation F1 |
| --- | --- | --- | --- |
| A0 | pretrain | 7 | 0.7423 |
| A1 | pretrain | 1 | 0.7260 |
| A2 | pretrain | 1 | 0.7106 |
| A3 | pretrain | 4 | 0.7041 |
| A4 | distill_0 | 1 | 0.5961 |

## Evaluation and metrics

Every model is evaluated unchanged on the same historical 500 answerable SQuAD v1.1 validation questions from 36 contexts, at 5%, 10%, ..., 50% removal. The primary matched-budget comparison is 40%; other levels assess transfer beyond the single training budget, not multi-budget training. Means weight questions equally. Each context-budget compression is reused across that context's questions. Deployment uses context and removal fraction only, without evaluation questions or answers.

F1 measures normalized answer-token overlap. Semantic similarity is answer embedding cosine similarity using all-MiniLM-L6-v2. Binary accuracy follows the existing rule: normalized exact equality, reference substring in the prediction, F1 >= 0.5, OR semantic similarity >= 0.75. It is not benchmark Exact Match or human-audited correctness. All variants retain these definitions.

## Primary results at 40% removal

| id | accuracy | f1 | semantic_similarity | actual_token_reduction |
| --- | --- | --- | --- | --- |
| A0 | 0.8120 | 0.7498 | 0.8506 | 0.3972 |
| A1 | 0.8160 | 0.7387 | 0.8510 | 0.3972 |
| A2 | 0.8160 | 0.7376 | 0.8476 | 0.3972 |
| A3 | 0.8300 | 0.7582 | 0.8536 | 0.3972 |
| A4 | 0.6360 | 0.5321 | 0.7265 | 0.3972 |

A0 reaches 81.2% accuracy and 0.7498 F1. Replacing the retention Transformer reduces mean F1 by 0.0111, and removing budget conditioning reduces it by 0.0121. Both variants nevertheless have slightly higher observed accuracy (81.6%). Removing structural features increases observed F1 by 0.0085 and accuracy by 1.8 percentage points. Thus the full architecture is not uniformly best on the final cohort.

A4 reaches 63.6% accuracy and 0.5321 F1: 17.6 percentage points lower accuracy and 0.2177 lower F1 than A0. This is evidence favoring the tested answer-span supervision procedure over the tested fresh reward-only procedure, not proof that all possible reward-based learning methods are inferior.

## Statistical comparisons at 40%

The estimand is mean paired question-level F1 difference (A0 minus each variant). Questions are paired by identifier and grouped by context. Percentile 95% confidence intervals resample 36 contexts with replacement 10,000 times, retaining all questions in each context and calculating question-weighted means. Two-sided context sign-flip tests use 50,000 random sign assignments and the plus-one Monte Carlo p-value correction. All question differences within a context receive the same sign. The test assumes independent contexts and sign symmetry under the null; it is not a paired t-test. Seed: 20261103. Holm correction covers the four planned primary comparisons at significance level alpha = 0.05. Confidence intervals are pointwise, not simultaneous familywise intervals.

Null hypothesis for each comparison: no difference in expected F1 between A0 and the corresponding variant. The table reports full-precision Monte Carlo p-values separately to avoid displaying small nonzero values as zero.

| Comparison | Mean F1 difference | 95% CI | Raw p | Holm p | Significant at 0.05? |
| --- | --- | --- | --- | --- | --- |
| A0 minus A1 | 0.011063 | [-0.029307, 0.053604] | 0.6108677826 | 1 | No |
| A0 minus A2 | 0.012114 | [-0.024890, 0.048105] | 0.5347093058 | 1 | No |
| A0 minus A3 | -0.008464 | [-0.052501, 0.029852] | 0.6944261115 | 1 | No |
| A0 minus A4 | 0.217672 | [0.160701, 0.269651] | 1.999960001e-05 | 7.999840003e-05 | Yes |

A1-A3: intervals include zero and Holm p-values equal 1.0. There is no statistically established advantage of A0 over these alternatives. This does not establish equivalence or prove that their removed components never help. A4: the interval [0.1607, 0.2697] excludes zero and Holm p is approximately 0.000080, supporting a substantial difference under this protocol.

## Complete cross-budget results

Values are on a 0-1 scale. Each cell is the mean of 500 question outcomes. These transfer-level tables are descriptive; no additional hypothesis tests are claimed at these budgets.

### Binary accuracy

| removal_percent | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.9000 | 0.9080 | 0.9040 | 0.9000 | 0.8880 |
| 10 | 0.9000 | 0.9040 | 0.9000 | 0.9000 | 0.8760 |
| 15 | 0.8700 | 0.8800 | 0.8860 | 0.8840 | 0.8760 |
| 20 | 0.8680 | 0.8620 | 0.8660 | 0.8620 | 0.8460 |
| 25 | 0.8680 | 0.8640 | 0.8420 | 0.8540 | 0.8200 |
| 30 | 0.8480 | 0.8500 | 0.8260 | 0.8580 | 0.7580 |
| 35 | 0.8340 | 0.8380 | 0.8340 | 0.8360 | 0.6940 |
| 40 | 0.8120 | 0.8160 | 0.8160 | 0.8300 | 0.6360 |
| 45 | 0.7960 | 0.7900 | 0.7840 | 0.8120 | 0.5940 |
| 50 | 0.7720 | 0.7720 | 0.7720 | 0.7960 | 0.5400 |

### Answer F1

| removal_percent | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.8626 | 0.8700 | 0.8707 | 0.8667 | 0.8488 |
| 10 | 0.8564 | 0.8690 | 0.8604 | 0.8604 | 0.8220 |
| 15 | 0.8244 | 0.8385 | 0.8421 | 0.8288 | 0.8150 |
| 20 | 0.8103 | 0.8155 | 0.8319 | 0.8072 | 0.7744 |
| 25 | 0.8139 | 0.8119 | 0.7995 | 0.8049 | 0.7308 |
| 30 | 0.7965 | 0.7997 | 0.7761 | 0.8058 | 0.6589 |
| 35 | 0.7667 | 0.7717 | 0.7641 | 0.7590 | 0.5939 |
| 40 | 0.7498 | 0.7387 | 0.7376 | 0.7582 | 0.5321 |
| 45 | 0.7334 | 0.7141 | 0.7073 | 0.7389 | 0.4839 |
| 50 | 0.6813 | 0.6998 | 0.6901 | 0.7084 | 0.4377 |

### Answer semantic similarity

| removal_percent | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.9178 | 0.9236 | 0.9199 | 0.9201 | 0.9103 |
| 10 | 0.9107 | 0.9198 | 0.9197 | 0.9144 | 0.8924 |
| 15 | 0.8922 | 0.9072 | 0.9036 | 0.9007 | 0.8899 |
| 20 | 0.8873 | 0.8897 | 0.8984 | 0.8873 | 0.8698 |
| 25 | 0.8928 | 0.8893 | 0.8794 | 0.8829 | 0.8440 |
| 30 | 0.8798 | 0.8772 | 0.8719 | 0.8836 | 0.7930 |
| 35 | 0.8573 | 0.8713 | 0.8668 | 0.8625 | 0.7573 |
| 40 | 0.8506 | 0.8510 | 0.8476 | 0.8536 | 0.7265 |
| 45 | 0.8396 | 0.8398 | 0.8398 | 0.8412 | 0.7066 |
| 50 | 0.8151 | 0.8251 | 0.8271 | 0.8296 | 0.6654 |

## Conclusions and limits

- Answer-span-supervised A0 clearly outperforms the tested reward-only A4 at the common 40% training budget. A4 changes the objective and uses additional search compute, so describe it as an alternative-supervision experiment.
- The results do not show a significant F1 benefit for the retention Transformer, explicit budget embedding, or structural features at 40%.
- A3 is slightly better on the observed test cohort, but its validation score is lower and its F1 advantage is not significant. A0 remains the validation-selected primary reference; do not call it the universally best model.
- Because all models train at one budget, A2 does not establish whether conditioning helps when learning several budgets jointly.
- Only one initialization seed is used. Context-resampling intervals do not quantify training-seed uncertainty. A1 also changes parameter count.
- The historical 500 questions were repeatedly inspected in previous thesis development. They are held out from this training process, but are not a pristine untouched discovery test. There are only 36 evaluation contexts.
- Permissive binary correctness thresholds can accept factually incorrect answers. F1 and semantic similarity provide additional, distinct evidence.

## Source records and plots

Paths below are relative to Codes/bsrt_neural_compression/results/controlled_ablation/.

- metrics.csv: this report filters ONLY A0-A4 (50 complete model-level records).
- paired_f1_tests_40.csv: four primary clustered comparisons.
- protocol.json: per-variant parameter counts and matched settings.
- A0/ through A4/: config.json, training_summary.json, checkpoints/, evaluation/final_500/bsrt_sample_results.csv, and comparison_summary.csv.
- primary_accuracy.pdf, primary_f1.pdf, primary_semantic_similarity.pdf: ONLY primary models; SVG and PNG copies are also available.
- Implementation: ../../controlled_ablation.py, ../../model.py, ../../training.py, and ../../evaluate.py.

This report is generated from completed records. It does not modify the thesis, recalculate experiments, or include supplementary-model results.
