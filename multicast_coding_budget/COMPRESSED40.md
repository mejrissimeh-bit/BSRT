# 40% aligned compression with budget-funded channel coding

This separate experiment compares original uncompressed token-ID unicast without
FEC against a single semantic-aligned compressed multicast, with a 40% token
removal target and LDPC protection. Previous results are not overwritten.

The cohort, questions, QPSK physical channels, actual receiver SNRs and physical
seeds are identical to `results/full`. There are 29 contexts, 87 K=2/3/5 groups,
145 distinct questions, two channels, six center SNRs and three seeds.
All 3,132 original-unicast cases are imported exactly from the completed database.
Only the 3,132 compressed-multicast cases need evaluating; exact received-text QA
cache hits are reused, and exact matching old waveforms/transports are checked.

## Semantic selection

For each receiver, reuse the saved QA-loss input-embedding gradient norms.
Normalize each receiver importance vector and form the shared angular-alignment
direction (normalized sum). Keep T-floor(0.4*T) highest-priority WordPieces in
original order. This is the earlier task-aware semantic alignment, NOT BSRT and
NOT a question-blind deployable compression policy. It uses receiver questions
and reference answers through the cached task gradients. No model is retrained.

27 contexts have reusable full-length gradient caches. The other two old caches
stop at 256 tokens although full contexts contain 285 and 418 tokens. Only their
five nested receiver tasks are recomputed on full contexts, and saved separately
under `full_length_relevance`. Context truncation is not counted as compression.
The achieved removal fraction can be slightly below 40% due to integer tokens.

## Budget and protection

T original tokens cost L=15*T bits; uncoded QPSK requires N0=L+(L mod 2).
The aggregate unicast budget is B=K*N0 bits, counting K complete copies.
Compressed multicast spends exactly B bits on one common encoded waveform.
Its effective information/wire rate is about 0.6/K: 0.30, 0.20 or 0.12.

Sionna 5G LDPC alone supports rates no lower than 0.2 in this implementation.
Mother-code length is capped accordingly. Any remaining wire budget repeats
mother-code bits cyclically; the decoder sums corresponding soft LLRs before
LDPC decoding. Therefore K=5 uses LDPC plus explicit repetition, not a native
rate-0.12 LDPC code. Minor rounding can require repetition at K=3 too. Repeated
bits are physical redundancy, never duplicate tokens in the reconstructed text.
Information blocks contain at most 254 tokens, and the existing Sionna zero-node
PCM-pruning workaround is reused. Every allocation records mother and wire
lengths, effective rate, repeated bits and actual token removal.

For example: 100 original tokens, 60 retained, K=5 gives B=7,500 bits.
The 900-bit compressed message is LDPC-encoded into 4,500 mother-code bits and
3,000 additional repeated coded bits. Original unicast sends 5*1,500 bits.

Both strategies have equal aggregate modeled transmitted bits/QPSK symbols and
unit-symbol-energy budget, not equal per-receiver energy or message duration.
Headers, scheduling, CRC, pilot overhead, CSI estimation and decoder costs remain
outside the model. Receiver offsets are +/-3 dB around group-center Es/N0;
AWGN and the existing normalized static TDL/ideal-CSI FFT-MMSE channel are reused.
The experiment cannot isolate compression from coding or multicast separately.

## Results and interpretation

Summary curves average receivers within each group, then contexts and seeds
within K, then K=2/3/5 equally. Results are descriptive; no significance is claimed.
Binary correctness uses the historical QA metric, not official Exact Match or
independent factual verification. High-SNR performance can be limited by removed
information even when LDPC completely removes channel bit errors.

Run or resume:

```powershell
python -u -m multicast_coding_budget.compressed40
```

Status and tables: `multicast_coding_budget/results/compressed40`.
On completion PDF/SVG/PNG comparisons are generated automatically in its `plots`.
Original results, thesis figures and the previous uncompressed-multicast run stay
unchanged. Source/configuration and cache fingerprints prevent incompatible resume.
