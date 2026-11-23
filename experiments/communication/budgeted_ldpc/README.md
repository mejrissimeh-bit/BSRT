# Reinvest compression savings in channel coding

This answers the supervisor's resource-allocation question, rather than simply
adding unrestricted rate-1/2 LDPC on top of compression.

If the original contains T0 token IDs and BSRT retains Tc IDs, use:

- Original information: L0 = 15*T0 bits.
- Original uncoded QPSK budget: B0 = 2*ceil(L0/2) bits, including modulation padding.
- Compressed information: K = 15*Tc bits.
- Rate-matched Sionna LDPC: k=K, n=B0; effective rate R=K/B0.

Every coded compressed frame uses exactly B0/2 QPSK symbols, the same as its
uncoded original. The saved source bits become redundancy, not extra traffic.
For actual 30% removal, R is approximately 0.7, not 0.5. Redundancy is 30% of
the original budget but 42.86% overhead relative to the compressed information.
Measured token counts determine the exact rate for every context/removal pair.

The first experiment uses AWGN at 0, 3, 5 and 7 dB, 200 questions from 36
contexts, 10/20/30/40% removal and seeds 42/123/2024. This gives 9,600 new
question/channel outcomes. The frozen BSRT A0 texts, original uncoded token-ID
outcomes and compressed uncoded outcomes are reused. No original simulation,
training or compression is repeated. The previously computed fixed-rate LDPC
outcomes are NOT equivalent and are not reused.

```powershell
python -u -m bsrt_budgeted_ldpc.run
```

Automatic checkpoint/resume uses the same command. To inspect progress:

```powershell
Get-Content ".\bsrt_budgeted_ldpc\results_awgn\run.log" -Tail 15 -Wait
Get-Content ".\bsrt_budgeted_ldpc\results_awgn\status.json"
```

After completion:

```powershell
python -m bsrt_budgeted_ldpc.run --plot
```

Separate accuracy and BER figures compare budgeted LDPC against the original
uncoded token-ID transmission. Budget tables and reports include the uncoded
BSRT comparison to distinguish coding benefit from compression alone. Existing
thesis figures are not overwritten. This test can show a benefit or a failure;
no outcome-aware rate tuning or guaranteed improvement is assumed.

Assumptions: a shared tokenizer; ideal payload length, code parameters and
framing knowledge, as in the baseline; no charged protocol/pilot/CRC overhead.
This is a controlled PHY/payload comparison, not a complete network protocol.
Both sides use QPSK and the same symbol energy. The equal symbol budget makes
their total frame energy equal at fixed SNR. No unequal token protection,
modulation switching, retransmission or interference model is introduced.

Validation:

```powershell
python -m unittest bsrt_budgeted_ldpc.test_budget
python -m bsrt_budgeted_ldpc.run --dry-run
python -m bsrt_budgeted_ldpc.run --smoke
```

The smoke test uses two questions from two contexts and SNR 0/7 dB, with all
four compression levels and all three seeds, in a separate folder.

## Multipath and packet-loss extension

The completed AWGN code and results remain frozen. Run both new channels with:

```powershell
python -u -m bsrt_budgeted_ldpc.run_extensions
```

The extension uses the same 200 questions / 36 contexts, 10/20/30/40% removal,
three seeds, and original per-context bit budget. TDL-A fading uses SNR
0/3/5/7 dB and the existing ideal-CSI FFT-MMSE receiver. Packet loss uses
0/0.01/0.02/0.05/0.1/0.2/0.3 over 256-bit packets, with erasure LLRs equal to
zero. It does not add AWGN to the packet-loss scenario or assume erased bits
were transmitted zeros. All retained information is LDPC-coded together.

There are 26,400 outcome slots: 9,600 fading and 16,800 packet-loss outcomes.
Neither original nor uncoded BSRT controls are recalculated. Completed QA metric
caches are imported read-only and reused only for the exact received text,
question, references and QA/metric schema. The shared prediction cache remains
active. Different channels are never substituted for one another just because
their nominal SNR matches. Identical packet erasure masks for an identical
codeword can reuse transport/decoding outputs even across nominal probabilities
and seeds; counts retain every planned question/seed outcome without pretending
they are independent received texts. Encoded codewords are also cached.

Outputs and separate accuracy/BER figures are in
`bsrt_budgeted_ldpc/results_fading_packet_loss/`. Original thesis figures and the
completed AWGN evidence report are not automatically overwritten.

```powershell
Get-Content ".\bsrt_budgeted_ldpc\results_fading_packet_loss\run.log" -Tail 15 -Wait
Get-Content ".\bsrt_budgeted_ldpc\results_fading_packet_loss\status.json"
python -m bsrt_budgeted_ldpc.run_extensions --plot
```

Progress JSON includes saved outcomes by channel and current-invocation cache
hits/new QA forward calls. Resume with the same run command; do not start a
second copy while one is running. Only after all expected outcomes are present
are the final comparison plots generated.

Validation:

```powershell
python -m unittest bsrt_budgeted_ldpc.test_extensions bsrt_budgeted_ldpc.test_budget
python -m bsrt_budgeted_ldpc.run_extensions --dry-run
python -m bsrt_budgeted_ldpc.run_extensions --smoke
```

The extension smoke run uses two questions / two contexts, 0/7 dB and loss
0/0.2, all four levels and all three seeds (96 outcomes), in a separate folder.

## Physical channels: 0.5 dB SNR sweep

The separate dense-grid runner keeps the completed experiments unchanged:

```powershell
python -u -m bsrt_budgeted_ldpc.run_snr_half_db
```

It sweeps AWGN and TDL-A from 0 to 7 dB inclusive in 0.5 dB steps, with the
same 200 questions / 36 contexts, 10/20/30/40% BSRT removal, three seeds and
equal-original-budget LDPC. No packet-loss experiment or compression training
is scheduled. Existing 0/3/5/7 dB observations are copied and checked, not
resimulated or interpolated: 19,200 coded plus 4,800 original outcomes.
The missing grid requires 52,800 coded and 13,200 original question/seed
outcomes. An original context is transmitted once per channel/SNR/seed,
shared across all associated questions and compression comparisons.

Output: `bsrt_budgeted_ldpc/results_snr_half_db/`. Its metrics, codewords,
transports and appended outcomes support resume with the same command.
Exact received-text/question/reference/schema metric hits are reused.
Progress is printed before each pending transmission; completed conditions
do not regenerate summaries. Original control points at new SNRs are actually
simulated, not interpolated from the earlier curve. Source fingerprints and
all imported scientific values are validated on resume. Do not run a second
worker against the same output directory.

```powershell
python -m bsrt_budgeted_ldpc.run_snr_half_db --dry-run
python -m bsrt_budgeted_ldpc.run_snr_half_db --smoke
python -m unittest bsrt_budgeted_ldpc.test_snr_half_db
Get-Content ".\bsrt_budgeted_ldpc\results_snr_half_db\status.json"
Get-Content ".\bsrt_budgeted_ldpc\results_snr_half_db\run.log" -Tail 10 -Wait
python -m bsrt_budgeted_ldpc.run_snr_half_db --plot
```

The smoke output is independent, using two questions from two contexts and
SNR 0/0.5/3.5/7 dB with all targets and seeds. Final combined AWGN/fading
accuracy and BER figures are written only after all planned records exist.
They retain the legend `Original context`; SNR samples are every 0.5 dB,
with integer tick labels and minor half-dB ticks for readability. Earlier
thesis figures/reports are not automatically replaced by incomplete results.
