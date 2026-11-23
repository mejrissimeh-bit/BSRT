# Verification of the latest non-singleton multicast comparison

Overall outcome: **all checks passed**.

Scope: 416 questions, 30 contexts, 109 non-singleton clusters; 29,952 paired receiver outcomes.
Source records were read only. Replays did not consume the experimental transport/QA caches.

## Checks

| Check | Result |
|---|---|
| Locked implementation files unchanged | PASS |
| Locked software versions match | PASS |
| Eligibility matches original non-singleton cluster memberships | PASS |
| Complete and unique paired subgroup outcomes | PASS |
| Identical receiver IDs, physical seeds, SNRs and source lengths across strategies | PASS |
| All receiver SNR offsets and seeds independently reconstructed | PASS |
| Exact equal subgroup wire budget | PASS |
| Frame dimensions cover every retained source bit within budget | PASS |
| BER and message-success labels match error counts | PASS |
| All 36,000 exported records agree with authoritative SQLite cases | PASS |
| Independent token-F1 recalculation for all 29,952 subgroup outcomes | PASS |
| Independent binary-label recalculation for all subgroup outcomes | PASS |
| Every plotted mean reproduced from original question-level records | PASS |
| Plot data exactly match recomputed means | PASS |
| All error-free messages reproduce clean text, predictions and all QA metrics | PASS |
| Fresh transport replay matches saved waveforms, received text and every recorded transport statistic | PASS |
| AWGN power, QPSK BER and soft-demapper sign sanity checks | PASS |
| Fresh local BERT inference reproduces selected saved predictions | PASS |
| Fresh Sentence-BERT scores and threshold decisions reproduce selected records | PASS |
| Original run lock, raw CSV and SQLite source files unchanged by audit | PASS |

## Interpretation and limitations

The full record checks cover cohort membership, aggregate budgets, pairing, error denominators, all F1/accuracy labels, plot means and error-free clean consistency.
Fresh transport replay covers 52 selected transmissions, not all 29,952 outcomes. Fresh QA covers eight predictions; fresh similarity checks cover 31 answer/reference pairs.
A successful replay is evidence of reproducibility, not an independent physical-channel implementation or proof that the model matches a real network.
All retained bits enter LDPC for included multicast clusters; source bits are used to compute evaluation error statistics, not as decoder inputs. Soft demapping uses received symbols and known channel/noise parameters.
The 0 dB x coordinate is a group center, with original receiver offsets within -3 to +3 dB. Code rates vary by cluster size and large clusters use repetition. High low-SNR QA is not a claim of universally error-free transport.
This remains a supervised task-aware, gold-answer-gradient reference, not BSRT or question-blind compression. Binary correctness uses permissive historical thresholds and is not manually verified factual correctness.
TDL is normalized/static with ideal CSI and approximate equalizer LLR variance. Headers, signaling, pilots, CRC, latency, energy at receivers and practical multicast feedback are not modeled.
Equal aggregate transmitted bits are verified; equal per-receiver energy or performance against all possible coded-unicast allocations is not established.
Excluding singletons conditions on a selected subgroup; it must not be reported as full-500 performance. No new significance claim is made.

## Evidence

- `checks.json`: results and locked source hashes.
- `transport_replays.csv`: exact replay comparisons.
- `awgn_sanity.csv`: measured/theoretical physical sanity checks.
- `fresh_qa_checks.csv`: uncached BERT predictions.
- `fresh_similarity_checks.csv`: fresh Sentence-BERT comparisons.
