# Accuracy-difference confidence intervals: latest multicast versus unicast

Cohort: 416 questions / 30 contexts / 109 non-singleton clusters. Both strategies use the same questions and three seeds (42, 123, 2024).

Method: paired context-clustered percentile bootstrap with 50,000 replicates and bootstrap seed 20261106. Each replicate samples 30 contexts with replacement. Both strategies, every eligible question and all three channel seeds are retained within each sampled context.
The resampled accuracy difference is the sum of per-context question-level differences divided by the sum of sampled question counts. Thus unequal context sizes retain the original question-weighted estimand, rather than changing to an equal-context mean.
Intervals are pointwise 95% percentile intervals, conditional on the observed channel seeds. They are not simultaneous familywise intervals. No p-values or multiple-comparison-adjusted significance test were calculated.
The source accuracy labels use the original historical OR/equality/containment criterion, not strict AND or the assistant-reviewed false-positive corrections.

| Channel | Group-center SNR (dB) | Unicast accuracy (%) | Multicast accuracy (%) | Difference (pp) | Pointwise 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| awgn | 0 | 7.69 | 73.72 | +66.03 | [+60.88, +70.53] |
| awgn | 3 | 23.16 | 87.74 | +64.58 | [+60.47, +68.23] |
| awgn | 5 | 45.19 | 90.38 | +45.19 | [+40.76, +49.00] |
| awgn | 7 | 67.15 | 90.38 | +23.24 | [+19.60, +26.55] |
| awgn | 10 | 87.90 | 90.38 | +2.48 | [+0.45, +4.40] |
| awgn | 15 | 91.59 | 90.38 | -1.20 | [-3.56, +1.09] |
| fading | 0 | 7.13 | 74.76 | +67.63 | [+61.46, +73.09] |
| fading | 3 | 22.84 | 87.42 | +64.58 | [+60.50, +68.02] |
| fading | 5 | 41.99 | 90.46 | +48.48 | [+44.04, +52.54] |
| fading | 7 | 64.34 | 90.38 | +26.04 | [+22.42, +29.33] |
| fading | 10 | 86.46 | 90.38 | +3.93 | [+1.56, +6.49] |
| fading | 15 | 91.59 | 90.38 | -1.20 | [-3.56, +1.09] |

## Limitations

These intervals quantify between-context sampling uncertainty conditional on this selected subgroup, its task-aware clustering, fixed seed outcomes, simulator assumptions and metric definition. They do not quantify annotation bias, training-selection bias, model mismatch, all possible channel realizations or uncertainty from assistant factual review.
An interval containing zero does not establish equivalence. An interval excluding zero at one point is not a multiple-comparison-adjusted claim across the complete SNR sweep.
No original outcomes, predictions, plots or experiment reports were overwritten.

Reproduce: `python -m task_aligned_multicast_500.bootstrap_accuracy_difference`.
