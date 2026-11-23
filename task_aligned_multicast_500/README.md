# Task-aligned multicast: all 500 final receiver questions

This independent experiment implements the confirmed comparison:

- Unicast sends one individually selected compressed token-ID message per question,
  with no FEC.
- Multicast sends one task-aligned compressed token-ID message per receiver cluster,
  spending the avoided duplicate transmissions on LDPC and, if needed, repetition.
- Both use the SAME aggregate budget: at most 60% of full-context token-ID unicast.

## Cohort and semantic selection

All 500 final evaluation questions across all 36 contexts are included. Each
context has K equal to its actual question count (2--29), not an artificial
K=2/3/5 subset. Every question is a separate receiver. Final means give equal
weight to every question, followed by averaging the three channel seeds. Large
contexts are NOT weighted equally with small ones.

The earlier task-alignment clustering procedure is reused: normalized receiver
QA-gradient importance vectors, cosine kernel, average-linkage agglomerative
clustering, 1--min(6,K) candidate clusters, minimum alignment 0.75 and representation
penalty 0.08. Co-priorities within a cluster minimize cosine misalignment. With
15 bits per token ID, importance-per-cost equals gradient importance divided by
15, not the old variable UTF-8 token cost. Individual unicast ranks each question's
own gradient importance. Cluster multicast ranks the cluster co-priority. Both
keep identical numbers of tokens, restoring their original ordering.

This is task-aware, answer-supervised reference alignment, NOT a question-blind
deployable BSRT policy. No model is retrained and no observed channel QA outcomes
are used to pick clusters. Each context's full original tokenization is used.
Old gradients are reused for the 32 contexts not truncated in the old cache. Four
old contexts stop at 256 tokens although full lengths are 285--418. Previously
extended compatible question gradients are reused; only missing full-length
gradients are calculated and separately cached. Thus neither hidden truncation
nor the earlier UTF-8 budget is silently counted as 40% token-ID compression.

## Exact budget

For T original WordPieces, an uncoded original QPSK frame costs
N0=2*ceil(15*T/2) bits. Each receiver's budget is
b=2*floor(0.6*N0/2). Retain n=floor(b/15) tokens. All unused sub-token capacity
is explicit zero padding, not additional information or error correction.
The context-wide budget is B=K*b <= 0.6*K*N0 bits.

Unicast sends K separate frames of b bits. A multicast cluster of m receivers
sends one frame of m*b bits. Summing over clusters gives exactly K*b in both
scenarios. Every cluster's source message has 15*n information bits.
Shared-frame effective rate is approximately 1/m, not 1/K unless all receivers
belong to a single cluster. A singleton has no sharing saving and stays uncoded.

For m>=2, use rate-matched Sionna 5G LDPC, splitting source blocks at 254 tokens.
LDPC mother-code rate cannot be below 0.2 here. Remaining cluster budget repeats
coded bits cyclically, and corresponding received soft LLRs are summed before
LDPC decoding. Clusters larger than five therefore generally need explicit
repetition. Rate, mother-code dimensions, repeated bits and cluster membership
are recorded. Source tokens are never repeated in the recovered text.

For example, T=100 gives N0=1500, b=900 bits and n=60 tokens. A context with ten
questions has B=9000 bits. Unicast uses ten 900-bit frames. If clusters have sizes
4,3,2,1, multicast uses four frames of 3600,2700,1800,900 bits, respectively;
the singleton remains uncoded and both scenarios use 9000 bits total.

## Channel and cache reuse

QPSK, AWGN and the existing normalized static TDL/ideal-CSI FFT-MMSE simulation;
group-center Es/N0 of 0,3,5,7,10,15 dB; seeds 42,123,2024. Within each full
context group, receiver SNR offsets span -3 to +3 dB. Receiver noise seeds match
between strategies. The waveform is common only within a multicast cluster.
All other receivers decode their own independently perturbed observation.

Earlier original-uncompressed or global-aligned transmission outcomes cannot be
substituted as the new baseline: their payloads/groups differ. Exact waveform,
coding, block, SNR and seed matches can reuse transport caches; exact received
text/question/reference/QA-setting matches reuse metric caches. The existing QA
prediction cache is also reused. New cases are committed atomically to SQLite;
resume skips completed cases. Config, code, software, payload and gradient-cache
fingerprints prevent incompatible resume. Previous experiments are untouched.

## Interpretation and outputs

Clean compressed QA performance is saved separately for each strategy. Error-free
decoded messages must match those predictions. The earlier approximately 91%
task-alignment result is not guaranteed: this experiment uses a different cost
definition, full contexts and a strict 60% budget with integer/QPSK rounding.
Singleton clusters and weak receivers can prevent a flat high-SNR plateau.

Binary accuracy retains the historical normalized-reference/F1/similarity
criterion, not official Exact Match or independent factual verification.
Results are descriptive; no confidence interval or significance claim is made.
Equal aggregate QPSK bits/symbol energy does not equalize per-receiver energy or
latency. Headers, cluster-membership signaling, CRC, pilots, scheduling, receiver
CSI estimation and decoder cost are not modeled.

Full run: 2,592 context-condition-strategy cases and 36,000 receiver outcomes.
No previous experiment or thesis figure is overwritten.

```powershell
python -u -m task_aligned_multicast_500.run
```

The output directory is `task_aligned_multicast_500/results/full`. It includes
`status.json`, `receiver_results.csv`, `summary_all_questions.csv`,
`clean_summary.csv`, `budget_allocations.json`, and a full provenance lock.
On completion, the automatic audit checks every paired receiver/seed/budget and
clean reconstruction; question-weighted PDF/SVG/PNG comparisons are generated
in `plots`. Resume with the same command. Use `--dry-run` to inspect the plan
(this may construct any missing full-length gradient caches).
