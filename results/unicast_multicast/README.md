# Task-aligned Unicast versus Multicast

This folder contains the final paired analysis of individually compressed Unicast and task-aligned Multicast under the same aggregate token-ID wire budget.

## Final figure cohort

- 416 receiver questions
- 30 contexts
- 109 task-alignment clusters with at least two receivers
- Three channel seeds
- AWGN and normalized TDL multipath fading
- Group-center SNRs from 0 to 15 dB
- 29,952 paired receiver outcomes

Both strategies use 524,268 wire bits per channel, SNR, and seed condition. Unicast sends one individually selected representation per receiver. Multicast sends one task-aligned representation per cluster and reinvests the avoided duplicate transmissions in LDPC and repetition.

`REPORT.md` defines the complete method and its boundaries. `summary_all_questions.csv` contains the plotted means. `confidence_intervals/` contains paired context-clustered intervals. `verification/` records the independent checks. `full_cohort/` preserves the original 500-question analysis for transparency.

This is an answer-supervised, task-aware reference experiment. It is separate from the deployable question-independent BSRT compressor.
