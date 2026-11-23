# Equal-budget LDPC result verification

## Audit outcome

All 30 recorded checks passed. Complete: 9,600 outcomes, 200 questions, 36 contexts, four removal levels, four SNRs and three seeds. No simulation source or recorded source dataset changed.

The raw records, exact comparison keys, all bit and symbol budgets, parity accounting, question/context identities, seeds and question-independent broadcasts were checked. All 9,600 F1 values and binary labels were independently recalculated from recorded predictions. All summary means were recalculated from raw rows. Sixteen coded transmissions plus their sixteen original counterparts were replayed at the shortest and longest contexts, reproducing stored BER and received IDs/text. Four fresh QA model predictions and semantic-similarity evaluations were independently checked. This is a targeted replay, not a full independent rerun.

## Accuracy

Question- and seed-weighted means, on a 0-1 scale:

| SNR_dB | original_uncoded | 10 | 20 | 30 | 40 |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0583 | 0.0400 | 0.0217 | 0.0283 | 0.0350 |
| 3.0000 | 0.1450 | 0.2050 | 0.1767 | 0.2800 | 0.8150 |
| 5.0000 | 0.3983 | 0.3800 | 0.7933 | 0.8600 | 0.8250 |
| 7.0000 | 0.7350 | 0.9217 | 0.8950 | 0.8600 | 0.8250 |

## Why the curves cross

Compression removes task information, but makes room for parity within the SAME original QPSK budget. More removal lowers the code rate and improves noise tolerance, at the cost of a lower clean-QA ceiling. At 3 dB the 40% variant has enough redundancy to nearly reach its clean compressed ceiling; at 5 dB the 30% variant preserves more information while still decoding all measured information bits correctly; at 7 dB the 10% variant benefits from retaining more source information. These are descriptive best points among the tested settings, not a validated adaptive online selection policy.

At 0 dB every coded variant underperforms the uncoded original. At 5 dB, the 10% variant also underperforms both the original and uncoded BSRT. Too little redundancy or a channel below the decoder's useful operating range can lead to failed decoding and a post-decoding BER larger than raw channel BER. LDPC does not guarantee recovery.

## Channel sanity check

Each actual context broadcast counted once (not once per question). The uncoded Gray-QPSK analytical bit-error probability at unit symbol energy is Q(sqrt(Es/N0)) = 0.5*erfc(sqrt(10^(SNR_dB/10)/2)). It matches raw coded-bit errors closely; coding changes post-decoding information BER, not the underlying noise.

| channel_value | measured_raw_coded_ber | qpsk_theoretical_ber |
| --- | --- | --- |
| 0.0000 | 0.1577 | 0.1587 |
| 3.0000 | 0.0794 | 0.0789 |
| 5.0000 | 0.0380 | 0.0377 |
| 7.0000 | 0.0126 | 0.0126 |

## Seed variation

| removal_percent | channel_value | 42 | 123 | 2024 |
| --- | --- | --- | --- | --- |
| 10 | 0.0000 | 0.0650 | 0.0300 | 0.0250 |
| 10 | 3.0000 | 0.2100 | 0.1950 | 0.2100 |
| 10 | 5.0000 | 0.3950 | 0.3550 | 0.3900 |
| 10 | 7.0000 | 0.9150 | 0.9300 | 0.9200 |
| 20 | 0.0000 | 0.0200 | 0.0150 | 0.0300 |
| 20 | 3.0000 | 0.1900 | 0.1500 | 0.1900 |
| 20 | 5.0000 | 0.8250 | 0.7450 | 0.8100 |
| 20 | 7.0000 | 0.8950 | 0.8950 | 0.8950 |
| 30 | 0.0000 | 0.0250 | 0.0300 | 0.0300 |
| 30 | 3.0000 | 0.3050 | 0.2400 | 0.2950 |
| 30 | 5.0000 | 0.8600 | 0.8600 | 0.8600 |
| 30 | 7.0000 | 0.8600 | 0.8600 | 0.8600 |
| 40 | 0.0000 | 0.0300 | 0.0300 | 0.0450 |
| 40 | 3.0000 | 0.8100 | 0.8200 | 0.8150 |
| 40 | 5.0000 | 0.8250 | 0.8250 | 0.8250 |
| 40 | 7.0000 | 0.8250 | 0.8250 | 0.8250 |

## Context-clustered uncertainty

Paired accuracy differences: budgeted LDPC minus uncoded original. Percentile 95% CIs use 10,000 bootstrap resamples of the 36 contexts, retaining every question and all three seeds within each selected context. Ratios of summed question-level differences to summed question counts preserve the original question-weighted estimand. Seed 20261104 controls bootstrap reproducibility. These are pointwise CIs, conditional on the three tested channel seeds; they are not simultaneous intervals, a familywise significance test or comprehensive uncertainty over arbitrary channels.

| removal_percent | snr_db | accuracy_difference | ci95_low | ci95_high |
| --- | --- | --- | --- | --- |
| 10 | 0.0000 | -0.0183 | -0.0382 | 0.0000 |
| 10 | 3.0000 | 0.0600 | 0.0131 | 0.1094 |
| 10 | 5.0000 | -0.0183 | -0.0626 | 0.0281 |
| 10 | 7.0000 | 0.1867 | 0.1363 | 0.2418 |
| 20 | 0.0000 | -0.0367 | -0.0599 | -0.0151 |
| 20 | 3.0000 | 0.0317 | -0.0220 | 0.0886 |
| 20 | 5.0000 | 0.3950 | 0.3221 | 0.4680 |
| 20 | 7.0000 | 0.1600 | 0.1095 | 0.2160 |
| 30 | 0.0000 | -0.0300 | -0.0539 | -0.0066 |
| 30 | 3.0000 | 0.1350 | 0.0804 | 0.1900 |
| 30 | 5.0000 | 0.4617 | 0.4000 | 0.5185 |
| 30 | 7.0000 | 0.1250 | 0.0660 | 0.1823 |
| 40 | 0.0000 | -0.0233 | -0.0469 | 0.0000 |
| 40 | 3.0000 | 0.6700 | 0.5871 | 0.7465 |
| 40 | 5.0000 | 0.4267 | 0.3643 | 0.4874 |
| 40 | 7.0000 | 0.0900 | 0.0340 | 0.1463 |

## Metric corroboration

| removal_percent | channel_value | accuracy | f1 | semantic_similarity | decoded_information_ber |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.0000 | 0.0400 | 0.0260 | 0.2807 | 0.1930 |
| 10 | 3.0000 | 0.2050 | 0.1509 | 0.4178 | 0.1175 |
| 10 | 5.0000 | 0.3800 | 0.3035 | 0.5467 | 0.0680 |
| 10 | 7.0000 | 0.9217 | 0.7982 | 0.9187 | 0.0007 |
| 20 | 0.0000 | 0.0217 | 0.0173 | 0.2769 | 0.1863 |
| 20 | 3.0000 | 0.1767 | 0.1324 | 0.4040 | 0.1016 |
| 20 | 5.0000 | 0.7933 | 0.6922 | 0.8328 | 0.0075 |
| 20 | 7.0000 | 0.8950 | 0.7813 | 0.8994 | 0.0000 |
| 30 | 0.0000 | 0.0283 | 0.0118 | 0.2614 | 0.1844 |
| 30 | 3.0000 | 0.2800 | 0.2144 | 0.4875 | 0.0634 |
| 30 | 5.0000 | 0.8600 | 0.7639 | 0.8954 | 0.0000 |
| 30 | 7.0000 | 0.8600 | 0.7639 | 0.8954 | 0.0000 |
| 40 | 0.0000 | 0.0350 | 0.0245 | 0.2689 | 0.2018 |
| 40 | 3.0000 | 0.8150 | 0.7156 | 0.8575 | 0.0008 |
| 40 | 5.0000 | 0.8250 | 0.7228 | 0.8622 | 0.0000 |
| 40 | 7.0000 | 0.8250 | 0.7228 | 0.8622 | 0.0000 |

Uncoded original metrics:

| channel_value | accuracy | f1 | semantic_similarity |
| --- | --- | --- | --- |
| 0.0000 | 0.0583 | 0.0359 | 0.3111 |
| 3.0000 | 0.1450 | 0.1166 | 0.3930 |
| 5.0000 | 0.3983 | 0.3088 | 0.5445 |
| 7.0000 | 0.7350 | 0.6137 | 0.7835 |

## Interpretation and limits

The supported conclusion is that, for this tested AWGN/QPSK setup, allocating compression savings to rate-matched LDPC can substantially improve receiver QA at the same transmitted-bit, symbol and unit-symbol-energy frame budget as uncoded original token-ID transmission. Budget equality holds per context/condition, not merely on average. This is a joint compression-and-coding benefit, not proof that compression alone makes the physical channel better or that the scheme is globally optimal.

The reference is deliberately uncoded original text represented by token IDs. No conclusion of superiority over every possible coded-original or source-coding scheme follows. Length/code parameters/framing are assumed ideally known; physical protocol headers, pilots, CRC and synchronization costs are not charged. Computational decoder energy/latency is not included. AWGN only: no fading/packet-loss generalization has been measured here. The 200-question/36-context subset has been used repeatedly during development and is not a pristine final holdout. Binary accuracy retains the historical permissive F1>=0.5 OR similarity>=0.75 OR normalized equality/substring criterion; it is not manual factual correctness. F1 and similarity corroborate the main improvements, but threshold sensitivity remains relevant. Measured zero BER means zero observed errors in these runs, not guaranteed future recovery.

## Detailed artifacts

verification_checks.json; verification_replayed_transmissions.csv; verification_context_bootstrap.csv; verification_ber_diagnostics.csv. Original result files and thesis figures were not altered by this audit.
