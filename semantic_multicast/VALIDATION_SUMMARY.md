# Implementation and evaluation status

## Completed and checked

- Reused the existing multi-receiver gradient cache, angular alignment, token-ID transport,
  QPSK/Sionna AWGN/TDL, variable-dimension LDPC and QA pipeline. No model was trained.
- Nine tests passed, including actual Sionna transmissions, a QPSK BER-scale check at
  0 dB, repeat-seed reproducibility, distinct receiver noise, immutable shared waveforms,
  and exact aggregate resource constraints.
- **Smoke:** one context, two receivers, 30% target, centers 3/15 dB, seed 42,
  AWGN and fading, uncoded and LDPC, both SNR profiles and both budget comparisons.
  All **64 cases / 128 receiver outcomes** completed and passed the saved-record audit.
- **Pilot:** two contexts, eight distinct questions, nested K=2/3/5 groups, 10/40%
  targets, centers 3/7 dB, seeds 42/123/2024, AWGN with rate-1/2 LDPC,
  both SNR profiles and both resource comparisons. All **480 cases / 1,440 receiver
  outcomes** completed and passed the independent audit. One context has only three
  questions, so K=5 is evaluated on one context, not two.
- Re-running the completed pilot confirmed that no channel or QA evaluations were repeated.
- Pilot figures were rendered and visually checked. Numerical results are in
  `results/pilot/`; smoke results are separate in `results/smoke/`.

## One actual pilot example

Context `00527421868151f4`, K=5, AWGN, 40% target removal, homogeneous 7 dB,
averaged across the three channel seeds. The source has 91 WordPieces; the shared
message keeps 55 (39.56% actual removal after integer rounding).

| Comparison | Scenario | Tokens per message | Aggregate QPSK symbols | Mean accuracy | Mean token F1 |
|---|---|---|---:|---:|---:|
| Equal message budgets | Unicast | 55 each | 4,125 | 1.000 | 0.9333 |
| Equal message budgets | Multicast | 55 shared | 825 | 1.000 | 0.8667 |
| Equal aggregate budget | Unicast | 11 each | 825 | 1.000 | 0.6500 |
| Equal aggregate budget | Multicast | 55 shared | 825 | 1.000 | 0.8667 |

The first comparison demonstrates **80% fewer modeled symbols**, with lower F1 from
shared task prioritization. The second spends the **same aggregate symbols** and
demonstrates why a shared, less compressed context can preserve more answer overlap.
These are different fairness questions, not interchangeable claims of superiority.

At the same removal/K with heterogeneous center 3 dB (receiver SNRs 0/1.5/3/4.5/6 dB),
the multicast mean accuracy is 0.800, and the observed probability that **every**
receiver bit-perfectly decodes is 0.000. At homogeneous 7 dB it is 1.000.
Success at stronger receivers therefore does not imply success for the entire group.

Accuracy follows the historical permissive OR criterion; it is not exact-match accuracy.
The equal-aggregate example's perfect binary score alongside only 0.650 F1 is one reason
to retain F1 and semantic similarity, rather than interpret accuracy alone.

These pilot contexts do not establish statistical significance or broad benchmark gains.
Savings at fixed equal message lengths are exactly 1-1/K (50%, 66.67%, 80%), verified
against actual coded arrays. They are not measured throughput or latency improvements.

## Important assumptions

This experiment uses cached **task-aware QA-gradient reference priorities**, including
reference-answer supervision and the original estimator's fallback behavior. It is not
a question/answer-blind BSRT inference experiment. No claims of question-blind deployment
are made. Token costs are now uniformly 15 bits, rather than the older UTF-8 density costs.

Only modeled source bits, FEC and modulation padding are counted. Framing, routing and
code parameters are assumed known; headers, CRC, pilots, retransmissions and computational
cost are not modeled. All links use the same fixed coding rate; link adaptation is absent.
The TDL receiver reuses the existing ideal-CSI FFT-MMSE and residual-ISI approximation.

## Full sweep: configured, NOT executed

The default configuration was dry-run checked: 36 contexts, 100 nested receiver groups
(36 with K=2, 35 with K=3, 29 with K=5), 165 distinct selected questions from the cached
500-question cohort, four removal targets, six SNR centers, two physical channels,
two coding modes, two profiles, two budget comparisons and three seeds.

It would produce **230,400 cases / 741,888 receiver outcomes**. The presence of
`results/full/plan.json` does NOT mean this experiment ran. This is deliberately left
as an explicit long-run command; inspect/reduce the JSON grid before launching it.

```powershell
python -m semantic_multicast.run --dry-run
python -m semantic_multicast.run
```

Primary evidence: `results/pilot/REPORT.md`, `summary.csv`, `receiver_results.csv`,
`paired_comparison.csv`, `verification.json`, and `plots/`. See README.md for all commands.
