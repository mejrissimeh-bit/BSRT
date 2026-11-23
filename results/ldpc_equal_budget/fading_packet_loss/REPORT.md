# Equal-budget LDPC: multipath and packet-loss extension

Status: 26400/26400 outcomes. COMPLETE.

200 questions / 36 contexts; removal [10, 20, 30, 40]%; seeds [42, 123, 2024]; fading SNR [0, 3, 5, 7] dB; packet loss [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3].

The frozen BSRT sources, original token-ID controls and uncoded BSRT controls are reused. Each new retained sequence is encoded as ONE Sionna 5G NR LDPC codeword, k=15*retained IDs, n=2*ceil(15*original IDs/2), 12 decoding iterations. All retained bits are protected; this is not unequal protection of a further token subset. Coded bits and QPSK symbols equal the corresponding original uncoded frame individually, not merely on average.

## Channel and receiver assumptions

Fading exactly reuses the existing Sionna TDL-A physical channel and ideal-CSI FFT-MMSE receiver: 30 ns RMS delay spread, 3.5 GHz carrier, 1 MHz bandwidth, zero speed, normalized SISO channel, sampled lags 0-4. Five sampled taps are not five independently specified propagation paths. Decoder soft outputs use the existing average equalizer gain and effective variance from filtered noise plus a Gaussian residual-ISI approximation. This is not an OFDM/pilot-estimated receiver. QPSK has unit symbol energy, SNR=Es/N0.

Packet loss is the same CUSTOM independent erasure process over 256-bit transmitted packets (possibly shorter final packet). Surviving bits are perfect: no additional AWGN is added. Sionna supplies LDPC, not the packet-erasure process. Known erased coded-bit positions receive LLR=0, not a confident zero. Surviving bits receive signed LLR magnitude 20, as in the prior coded packet model. Decoded information IDs are unpacked normally; erased CODED positions are not misinterpreted as known erased information-token positions. Raw zero-filled BER is diagnostic, not the loss probability.

The original and coded frame have equal length, packet count and deterministic seed for a fixed context/loss probability/base seed, so their packet erasure masks match. The uncoded compressed control is shorter and need not have an identical mask. Lengths, code parameters, framing and erasure locations are assumed ideal known side information. Headers, pilots, CRC and synchronization overhead, decoder energy and latency are not simulated.

One reconstructed context is shared across all its questions. QA remains deepset/bert-base-cased-squad2 (384, stride 128, max answer 15, impossible answers off); F1 and all-MiniLM-L6-v2 answer cosine similarity use the same references. Binary accuracy uses normalized equality/reference substring OR F1>=0.5 OR similarity>=0.75, not a manual factual-correctness audit. Means weight questions and three seeds equally; shared-context questions are not independent contexts.

## Cache reuse

Existing metric entries are copied read-only into the extension metric cache. A hit requires the identical hash of QA/model settings, metric implementation, actual RECEIVED text, question and references. Merely matching nominal SNR, removal level or context does not permit reuse. The shared QA prediction cache provides a second reuse layer. AWGN transport outcomes are not substituted for fading or packet-loss outcomes.

Codewords are stored and reused for identical source IDs, dimensions and coding implementation. For packet loss, identical input/code settings and identical erased packet masks yield identical observations and deterministic decoder outputs, even when generated under different seeds/probabilities; these transports are reused without repeating decoding. Actual fading transports are keyed by channel settings and physical seed. Saved outcomes and all caches support resume.

Current invocation counters (not lifetime totals): {"codeword_hits": 5, "metric_hits": 6296, "new_codewords": 49, "new_metric_entries": 3232, "new_qa_forward_calls": 3232, "new_transports": 1097, "transport_hits": 667}

## Results

Comparisons below include only complete cells with all questions and seeds. Incomplete cells remain in summary.csv but are not compared against full-cohort controls.

| channel | removal_percent | channel_value | outcomes | accuracy | f1 | semantic_similarity | original_accuracy | uncoded_bsrt_accuracy | accuracy_gain_vs_original | decoded_information_ber | code_rate | transmitted_bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fading | 10 | 0.0000 | 600 | 0.0367 | 0.0183 | 0.2725 | 0.0500 | 0.0283 | -0.0133 | 0.1957 | 0.9042 | 2132.6800 |
| fading | 10 | 3.0000 | 600 | 0.1333 | 0.0991 | 0.3675 | 0.1917 | 0.1633 | -0.0583 | 0.1205 | 0.9042 | 2132.6800 |
| fading | 10 | 5.0000 | 600 | 0.3717 | 0.2882 | 0.5534 | 0.3983 | 0.3783 | -0.0267 | 0.0697 | 0.9042 | 2132.6800 |
| fading | 10 | 7.0000 | 600 | 0.8917 | 0.7701 | 0.8999 | 0.7367 | 0.7083 | 0.1550 | 0.0032 | 0.9042 | 2132.6800 |
| fading | 20 | 0.0000 | 600 | 0.0433 | 0.0258 | 0.2801 | 0.0500 | 0.0267 | -0.0067 | 0.1903 | 0.8029 | 2132.6800 |
| fading | 20 | 3.0000 | 600 | 0.1633 | 0.1155 | 0.3977 | 0.1917 | 0.1867 | -0.0283 | 0.1018 | 0.8029 | 2132.6800 |
| fading | 20 | 5.0000 | 600 | 0.7883 | 0.6659 | 0.8194 | 0.3983 | 0.4250 | 0.3900 | 0.0087 | 0.8029 | 2132.6800 |
| fading | 20 | 7.0000 | 600 | 0.8783 | 0.7649 | 0.8899 | 0.7367 | 0.6667 | 0.1417 | 0.0019 | 0.8029 | 2132.6800 |
| fading | 30 | 0.0000 | 600 | 0.0433 | 0.0220 | 0.2749 | 0.0500 | 0.0317 | -0.0067 | 0.1867 | 0.7043 | 2132.6800 |
| fading | 30 | 3.0000 | 600 | 0.2617 | 0.1985 | 0.4733 | 0.1917 | 0.1433 | 0.0700 | 0.0663 | 0.7043 | 2132.6800 |
| fading | 30 | 5.0000 | 600 | 0.8500 | 0.7539 | 0.8879 | 0.3983 | 0.3433 | 0.4517 | 0.0016 | 0.7043 | 2132.6800 |
| fading | 30 | 7.0000 | 600 | 0.8567 | 0.7596 | 0.8937 | 0.7367 | 0.6717 | 0.1200 | 0.0003 | 0.7043 | 2132.6800 |
| fading | 40 | 0.0000 | 600 | 0.0267 | 0.0145 | 0.2619 | 0.0500 | 0.0417 | -0.0233 | 0.2047 | 0.6028 | 2132.6800 |
| fading | 40 | 3.0000 | 600 | 0.8250 | 0.7133 | 0.8568 | 0.1917 | 0.1533 | 0.6333 | 0.0021 | 0.6028 | 2132.6800 |
| fading | 40 | 5.0000 | 600 | 0.8150 | 0.7128 | 0.8545 | 0.3983 | 0.3483 | 0.4167 | 0.0007 | 0.6028 | 2132.6800 |
| fading | 40 | 7.0000 | 600 | 0.8250 | 0.7228 | 0.8622 | 0.7367 | 0.6800 | 0.0883 | 0.0000 | 0.6028 | 2132.6800 |
| packet_loss | 10 | 0.0000 | 600 | 0.9300 | 0.8111 | 0.9257 | 0.9350 | 0.9300 | -0.0050 | 0.0000 | 0.9042 | 2132.6800 |
| packet_loss | 10 | 0.0100 | 600 | 0.9267 | 0.8065 | 0.9226 | 0.9233 | 0.9200 | 0.0033 | 0.0080 | 0.9042 | 2132.6800 |
| packet_loss | 10 | 0.0200 | 600 | 0.8883 | 0.7834 | 0.8962 | 0.9083 | 0.9033 | -0.0200 | 0.0267 | 0.9042 | 2132.6800 |
| packet_loss | 10 | 0.0500 | 600 | 0.8767 | 0.7529 | 0.8884 | 0.8967 | 0.8917 | -0.0200 | 0.0495 | 0.9042 | 2132.6800 |
| packet_loss | 10 | 0.1000 | 600 | 0.8433 | 0.7275 | 0.8666 | 0.8633 | 0.8567 | -0.0200 | 0.0705 | 0.9042 | 2132.6800 |
| packet_loss | 10 | 0.2000 | 600 | 0.6833 | 0.5850 | 0.7638 | 0.7283 | 0.7117 | -0.0450 | 0.1678 | 0.9042 | 2132.6800 |
| packet_loss | 10 | 0.3000 | 600 | 0.5883 | 0.4903 | 0.6938 | 0.6833 | 0.6617 | -0.0950 | 0.2263 | 0.9042 | 2132.6800 |
| packet_loss | 20 | 0.0000 | 600 | 0.8950 | 0.7813 | 0.8994 | 0.9350 | 0.8950 | -0.0400 | 0.0000 | 0.8029 | 2132.6800 |
| packet_loss | 20 | 0.0100 | 600 | 0.8867 | 0.7754 | 0.8932 | 0.9233 | 0.8883 | -0.0367 | 0.0066 | 0.8029 | 2132.6800 |
| packet_loss | 20 | 0.0200 | 600 | 0.8733 | 0.7613 | 0.8849 | 0.9083 | 0.8733 | -0.0350 | 0.0207 | 0.8029 | 2132.6800 |
| packet_loss | 20 | 0.0500 | 600 | 0.8633 | 0.7463 | 0.8762 | 0.8967 | 0.8533 | -0.0333 | 0.0335 | 0.8029 | 2132.6800 |
| packet_loss | 20 | 0.1000 | 600 | 0.8300 | 0.7263 | 0.8592 | 0.8633 | 0.8483 | -0.0333 | 0.0545 | 0.8029 | 2132.6800 |
| packet_loss | 20 | 0.2000 | 600 | 0.7017 | 0.5887 | 0.7719 | 0.7283 | 0.7117 | -0.0267 | 0.1552 | 0.8029 | 2132.6800 |
| packet_loss | 20 | 0.3000 | 600 | 0.5833 | 0.5010 | 0.6964 | 0.6833 | 0.6767 | -0.1000 | 0.2167 | 0.8029 | 2132.6800 |
| packet_loss | 30 | 0.0000 | 600 | 0.8600 | 0.7639 | 0.8954 | 0.9350 | 0.8600 | -0.0750 | 0.0000 | 0.7043 | 2132.6800 |
| packet_loss | 30 | 0.0100 | 600 | 0.8533 | 0.7586 | 0.8911 | 0.9233 | 0.8500 | -0.0700 | 0.0050 | 0.7043 | 2132.6800 |
| packet_loss | 30 | 0.0200 | 600 | 0.8417 | 0.7451 | 0.8801 | 0.9083 | 0.8400 | -0.0667 | 0.0124 | 0.7043 | 2132.6800 |
| packet_loss | 30 | 0.0500 | 600 | 0.8367 | 0.7410 | 0.8807 | 0.8967 | 0.8267 | -0.0600 | 0.0205 | 0.7043 | 2132.6800 |
| packet_loss | 30 | 0.1000 | 600 | 0.8050 | 0.7183 | 0.8639 | 0.8633 | 0.7917 | -0.0583 | 0.0451 | 0.7043 | 2132.6800 |
| packet_loss | 30 | 0.2000 | 600 | 0.6883 | 0.5964 | 0.7749 | 0.7283 | 0.6633 | -0.0400 | 0.1405 | 0.7043 | 2132.6800 |
| packet_loss | 30 | 0.3000 | 600 | 0.5833 | 0.5021 | 0.7018 | 0.6833 | 0.5817 | -0.1000 | 0.2023 | 0.7043 | 2132.6800 |
| packet_loss | 40 | 0.0000 | 600 | 0.8250 | 0.7228 | 0.8622 | 0.9350 | 0.8250 | -0.1100 | 0.0000 | 0.6028 | 2132.6800 |
| packet_loss | 40 | 0.0100 | 600 | 0.8250 | 0.7228 | 0.8622 | 0.9233 | 0.8167 | -0.0983 | 0.0000 | 0.6028 | 2132.6800 |
| packet_loss | 40 | 0.0200 | 600 | 0.8217 | 0.7161 | 0.8586 | 0.9083 | 0.8083 | -0.0867 | 0.0081 | 0.6028 | 2132.6800 |
| packet_loss | 40 | 0.0500 | 600 | 0.7950 | 0.6940 | 0.8454 | 0.8967 | 0.8017 | -0.1017 | 0.0261 | 0.6028 | 2132.6800 |
| packet_loss | 40 | 0.1000 | 600 | 0.7650 | 0.6614 | 0.8214 | 0.8633 | 0.7750 | -0.0983 | 0.0349 | 0.6028 | 2132.6800 |
| packet_loss | 40 | 0.2000 | 600 | 0.6583 | 0.5659 | 0.7423 | 0.7283 | 0.6533 | -0.0700 | 0.1244 | 0.6028 | 2132.6800 |
| packet_loss | 40 | 0.3000 | 600 | 0.5650 | 0.4909 | 0.6779 | 0.6833 | 0.5633 | -0.1183 | 0.2045 | 0.6028 | 2132.6800 |

No previous AWGN result, final thesis figure or combined thesis-writing report is overwritten. No claim of significance, superiority at all conditions, or a new adaptive compression policy follows from these means. The same repeatedly viewed 200-question/36-context cohort is used, not a pristine new holdout.
