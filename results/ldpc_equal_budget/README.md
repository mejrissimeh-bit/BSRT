# Equal-budget LDPC experiments

Semantic compression reduces the source payload. These experiments reinvest the saved bits in channel protection while preserving the original uncoded token-ID frame budget exactly for each context and condition.

## Included evidence

- `awgn/` contains the complete 9,600-outcome study, aggregate metrics, BER figures, a 30-check audit, replay checks, and context-clustered intervals
- `fading_packet_loss/` contains the TDL fading and packet-erasure extensions with their figures and aggregate tables
- `half_db/` contains the dense 0.5 dB AWGN and TDL sweep with 90,000 outcomes
- `budgeted_ldpc_awgn_fading_accuracy.png` is the final thesis comparison across AWGN and TDL fading

The principal constraint is equality of transmitted bits and QPSK symbols between each coded BSRT condition and the uncoded original-token reference. The comparison does not include protocol headers, pilots, synchronization, CRC, retransmission, or decoder energy.

The results do not claim universal LDPC improvement. All coded variants lose at 0 dB in the principal AWGN study, while different compression and code-rate points become preferable as SNR changes.
