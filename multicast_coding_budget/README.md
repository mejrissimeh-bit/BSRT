# Multicast savings fund LDPC on the original token payload

This is the requested **uncompressed, uncoded unicast versus uncompressed,
LDPC-protected multicast** comparison. Both scenarios contain exactly the same full
original tokenizer IDs. No 10% fallback, semantic token selection or model training is used.
It is separate from the previous compressed-versus-compressed alignment experiment.

For T tokens and K receivers:

- Information bits: L=15T.
- One uncoded QPSK frame: N0=2*ceil(L/2) bits.
- K-copy unicast, without coding: B=K*N0 transmitted bits.
- One multicast waveform uses at most B bits for LDPC, giving rate approximately 1/K.
- Rates are approximately 1/2, 1/3 and 1/5 for K=2,3,5. All original tokens are protected.

Long originals use balanced contiguous LDPC blocks of at most 254 tokens, avoiding
Sionna's automatic low-rate base-graph limits. At rate 1/5, integer modulation rounding
can leave a few bits unspent. No budget is exceeded. This is an aggregate resource
constraint, not equal per-receiver energy or latency. Headers/CRC/pilots are not modeled.

## Run from Codes

```powershell
python -m unittest multicast_coding_budget.test_transport -v
python -m multicast_coding_budget.run --smoke
python -m multicast_coding_budget.run --dry-run
python -m multicast_coding_budget.run
python -m multicast_coding_budget.run --plot-only
python -m multicast_coding_budget.audit --output multicast_coding_budget/results/full
```

The full run uses the 29 contexts common to K=2,3,5, 145 distinct questions, the saved
receiver groups, heterogeneous channels, QPSK, AWGN/TDL, center SNRs 0/3/5/7/10/15 dB,
and seeds 42/123/2024. Expected: 6,264 cases / 20,880 receiver outcomes. No redundant
removal sweep is needed: this experiment has 0% removal only. A smoke run uses one
context, all K, both channels, centers 0/3/7 dB and seed 42.

Existing received-text/question/reference/model/metric matches are reused read-only
from the previous multicast and original-token-ID QA caches. Shared QA prediction
caching is also reused. Earlier uncompressed transmissions have different physical
seeds/cohorts: their averaged curves are not substituted as paired outcomes. Missing
receiver channels are simulated. Completed cases and exact transports resume without
recomputation. Source payloads are verified by cohort context hashes and not truncated.

## Follow progress / outputs

```powershell
Get-Content .\multicast_coding_budget\results\full\status.json
Get-Content .\multicast_coding_budget\results\full\run_stdout.log -Tail 10 -Wait
```

The log exists when started with redirected output. Results: group_results.csv,
receiver_results.csv, summary_per_k.csv, summary_averaged_k.csv, budget_allocations.json,
run_lock.json, REPORT.md and plots/. Figures weight K equally over common contexts.
The report distinguishes modeled resources, bit-perfect decoding and historical QA
accuracy. Improvement at 3 dB is measured, not assumed. Old thesis plots are untouched.
