# Task-aware semantic multicast versus unicast

Independent experiment: no old results, thesis figures or trained models are overwritten.
Uses the final 500-question/36-context gradient cache. This is a question/answer-supervised
task-aware alignment reference, NOT a new question-blind BSRT model. No training is performed.

## Commands (from Codes)

```powershell
python -m unittest semantic_multicast.test_multicast -v
python -m semantic_multicast.run --smoke --dry-run
python -m semantic_multicast.run --smoke
python -m semantic_multicast.run --config semantic_multicast/config_pilot.json
python -m semantic_multicast.audit --output semantic_multicast/results/pilot
python -m semantic_multicast.run --dry-run
python -m semantic_multicast.run
python -m semantic_multicast.run --plot-only
```

The full grid is intentionally not launched automatically. Inspect its printed plan first.
Re-running the identical command resumes atomically committed cases; no duplicate outcomes.
Changing scientific settings or input implementations requires a different output_dir.
Use `--config` to configure contexts, K, removal, seeds, SNRs, channels, coding, budgets and profiles.
Smoke output: `semantic_multicast/results/smoke`; pilot: `results/pilot`.

## Focused continuation: heterogeneous receivers, equal aggregate budget

The broad full sweep was stopped at the user's request. Its original database remains intact.
The new entry point verifies and imports all compatible completed cases and exact channel/QA
caches, then computes only missing heterogeneous/equal-aggregate cases. Compatible pilot
and smoke results are imported too; duplicated cases are retained only once:

```powershell
python -m unittest semantic_multicast.test_focused -v
python -m semantic_multicast.run_focused --prepare-only
python -m semantic_multicast.run_focused
Get-Content semantic_multicast/results/heterogeneous_equal_budget/status.json
```

This keeps K=2/3/5, removal 10/20/30/40%, SNR 0/3/5/7/10/15 dB, both physical channels,
uncoded/LDPC and three seeds. Total: 57,600 cases / 185,472 receiver outcomes. The source
is `results/full`; the separate destination is `results/heterogeneous_equal_budget`.
Do not change the old locked run.py or edit the old full configuration to resume a different grid.
Importing calls neither Sionna channels nor QA. Completed case keys are skipped by the existing
runner; exact caches also reuse compatible received-text results from other conditions. Import is
idempotent and refuses changed physics, cohorts, software, implementations or conflicting cache
entries. Run the importer only when the destination experiment is not already running.

## Reused implementations

- `paper_multi_receiver_task_alignment/run_task_alignment.py`: cached QA-gradient relevance,
  normalization, projected angular co-priority. Gradients and original identities are checked.
- `bsrt_communication_evaluation/token_ids.py`: 15-bit encoding, invalid-ID handling and text decoding.
- `bsrt_communication_evaluation/channel.py`: QPSK map/demap, Sionna AWGN/TDL and FFT-MMSE receiver.
- `bsrt_budgeted_ldpc/channel.py`: variable-dimension Sionna LDPC encoder and decoder.
- `bsrt_neural_compression/reward.py`: existing QA and exact-context prediction cache.

`core.py` exposes a frame builder and a receiver separately: multicast encodes/modulates ONCE
outside receiver/channel loops. Receivers share the exact same immutable waveform but have
distinct noise/fading seeds, channel observations, decoders and QA outputs.

## Two different fairness questions

1. **Equal per-representation compression:** same retained token count; unicast transmits K
   different messages, multicast one. Actual savings are exactly `1-1/K` at the fixed rate.
2. **Equal aggregate resources:** unicast divides the shared message's coded-symbol budget
   between its K messages. Unicast therefore retains fewer tokens per receiver. Integer
   remainders are saved; no hidden budget excess is allowed.

Both use fixed common LDPC rate (default 1/2), not receiver-specific link adaptation.
Source payload is `15*T` bits; uncoded transmitted length is `2*ceil(15*T/2)`;
LDPC length is `2*ceil((15*T/R)/2)`. Multicast counts one waveform, unicast sums K.
Framing/code dimensions are assumed known. No headers, pilots, CRC, feedback, guard intervals,
retransmission, latency or scheduling are modeled. Do not claim overhead-complete network savings.

## Results

- `plan.json`, `groups.json`, `run_lock.json`: cohort, expected work, source/software fingerprints.
- `results.sqlite3`: transactional group and receiver records plus exact metric cache.
- `status.json`: progress; inspect with `Get-Content semantic_multicast/results/pilot/status.json`.
- `receiver_results.csv`: each question, received text, prediction, metrics, SNR, seeds and waveform hash.
- `group_results.csv`: one row per scenario/group/channel/seed/budget condition, resources counted once.
- `summary.csv`, `receiver_summary.csv`, `paired_comparison.csv`, `paired_summary.csv`.
- `REPORT.md`: setup, actual numerical differences, assumptions and limitations.
- `verification.json`: independent audit of saved rows (`python -m semantic_multicast.audit`).
- `plots/`: five requested plot families in PDF/SVG/PNG, separate budget/channel/coding conditions.

Receiver decode success is offline bit-perfect equality, not a receiver CRC. LDPC BLER uses
one codeword per message; uncoded frame errors are labeled separately. All-receiver success
is the observed conjunction. QA accuracy preserves the historical OR threshold definition;
see the generated report. Overlapping K groups and seeds are not independent contexts.
