# Compression savings reinvested in LDPC

Status: 9600/9600 new question/channel outcomes. COMPLETE.

## Question and resource constraint

Can semantic compression improve noisy-channel QA at no greater transmission cost by spending its saved bits on LDPC redundancy? For original token count T0 and compressed count Tc, L0=15*T0, K=15*Tc, and B0=2*ceil(L0/2). The Sionna 5G NR encoder is rate-matched to (k,n)=(K,B0). Every coded BSRT frame therefore uses EXACTLY the original uncoded QPSK budget B0 bits / B0/2 symbols. Equal symbol energy also means equal total frame energy at a given SNR. This is not the earlier padded 512/1024 rate-1/2 experiment and cannot reuse those coded outcomes.

For 30% actual token removal, R=K/B0 is approximately 0.7. Redundancy is approximately 30% of the ORIGINAL bit budget, or 0.3/0.7=42.86% overhead relative to the COMPRESSED information. Exact rates use measured token counts, not nominal removal targets. There is no additional application information padding; internal LDPC shortening/puncturing is handled by Sionna. Source length, code parameters and framing remain ideally known, just as in the cached controls. Protocol headers, pilots, synchronization, CRC and retransmissions are outside the model; no claim of an overhead-complete deployed protocol is made.

## Setup

200 questions / 36 contexts; removal targets [10, 20, 30, 40]%; AWGN SNR [0, 3, 5, 7] dB; three seeds [42, 123, 2024]. One frozen full-reference BSRT A0; compressed text and original/BSRT uncoded outcomes are reused from their completed token-ID experiments on the same questions. No original context transmission, compression or model training is repeated.

Sionna complex AWGN y=x+n, n~CN(0,N0), N0=10^(-SNR/10); unit-energy QPSK. SNR is Es/N0. The SAME physical channel and context-hashed seed rule as the uncoded controls are used, without coding/method/removal offsets. Rate-matched LDPC uses 12 soft-input decoding iterations. One decoded context is shared by all its questions. Equal original/coded symbol counts and seed conditions do not assert identical finite noise draws through every library invocation. No task outcome is used to choose the rate or allocation.

Receiver: 15-bit BERT IDs; invalid/reserved IDs become [UNK]; tokenizer decoding with space cleanup; deepset/bert-base-cased-squad2 QA, max sequence 384, stride 128, max answer 15. Actual decoded text is evaluated, with exact-text QA caching. Metrics retain historical token F1, all-MiniLM-L6-v2 cosine similarity, and binary correctness (normalized equality/reference substring OR F1>=0.5 OR similarity>=0.75). Accuracy is not exact match or manual factual verification. Question-weighted averages include three seeds; shared contexts are not independent significance-test replicates. No significant improvement is presumed.

## Results

| removal_percent | channel_value | outcomes | accuracy | original_accuracy | uncoded_bsrt_accuracy | accuracy_gain_vs_original | decoded_information_ber | code_rate | transmitted_bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.0000 | 600 | 0.0400 | 0.0583 | 0.0283 | -0.0183 | 0.1930 | 0.9042 | 2132.6800 |
| 10 | 3.0000 | 600 | 0.2050 | 0.1450 | 0.1800 | 0.0600 | 0.1175 | 0.9042 | 2132.6800 |
| 10 | 5.0000 | 600 | 0.3800 | 0.3983 | 0.4100 | -0.0183 | 0.0680 | 0.9042 | 2132.6800 |
| 10 | 7.0000 | 600 | 0.9217 | 0.7350 | 0.7683 | 0.1867 | 0.0007 | 0.9042 | 2132.6800 |
| 20 | 0.0000 | 600 | 0.0217 | 0.0583 | 0.0300 | -0.0367 | 0.1863 | 0.8029 | 2132.6800 |
| 20 | 3.0000 | 600 | 0.1767 | 0.1450 | 0.1583 | 0.0317 | 0.1016 | 0.8029 | 2132.6800 |
| 20 | 5.0000 | 600 | 0.7933 | 0.3983 | 0.3900 | 0.3950 | 0.0075 | 0.8029 | 2132.6800 |
| 20 | 7.0000 | 600 | 0.8950 | 0.7350 | 0.7300 | 0.1600 | 0.0000 | 0.8029 | 2132.6800 |
| 30 | 0.0000 | 600 | 0.0283 | 0.0583 | 0.0433 | -0.0300 | 0.1844 | 0.7043 | 2132.6800 |
| 30 | 3.0000 | 600 | 0.2800 | 0.1450 | 0.2000 | 0.1350 | 0.0634 | 0.7043 | 2132.6800 |
| 30 | 5.0000 | 600 | 0.8600 | 0.3983 | 0.3767 | 0.4617 | 0.0000 | 0.7043 | 2132.6800 |
| 30 | 7.0000 | 600 | 0.8600 | 0.7350 | 0.7050 | 0.1250 | 0.0000 | 0.7043 | 2132.6800 |
| 40 | 0.0000 | 600 | 0.0350 | 0.0583 | 0.0483 | -0.0233 | 0.2018 | 0.6028 | 2132.6800 |
| 40 | 3.0000 | 600 | 0.8150 | 0.1450 | 0.1800 | 0.6700 | 0.0008 | 0.6028 | 2132.6800 |
| 40 | 5.0000 | 600 | 0.8250 | 0.3983 | 0.3700 | 0.4267 | 0.0000 | 0.6028 | 2132.6800 |
| 40 | 7.0000 | 600 | 0.8250 | 0.7350 | 0.6550 | 0.0900 | 0.0000 | 0.6028 | 2132.6800 |

## Artifacts

budget_allocations.csv records each context/removal budget and rate. sample_results.csv records new budgeted-LDPC outcomes only; reference_controls.csv contains matched existing uncoded scores. summary.csv and per_seed_summary.csv retain means and seed variation; status.json gives the active progress target. run_lock.json fingerprints source records and implementation. Figures are generated only after all conditions are complete. Existing thesis figures are not overwritten.
