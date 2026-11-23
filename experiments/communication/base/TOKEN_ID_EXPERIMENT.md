# BSRT token-ID communication evaluation

## Current configuration: cached original only

The active default is now `config_token_ids_cached_original.json`. The previous
fresh-original-token-ID run was stopped at the user's request and preserved.
The revised run computes ONLY BSRT (uncoded/LDPC token IDs); original raw-text
scores come from the existing uncoded UTF-8 caches. Its output folder is
`results_token_ids_cached_original`. This is not a matched-representation or
equal-coding control; plots label the original curve explicitly as cached,
uncoded UTF-8 in both columns. No original QA or original transmissions run.

```powershell
Get-Content ".\bsrt_communication_evaluation\results_token_ids_cached_original\run.log" -Tail 15 -Wait
```

There are 91,200 BSRT outcomes to complete, with prior BSRT work imported and
QA/transport caches reused. `reuse_provenance.json` records the retained work.
The remainder below documents the earlier matched-original design; its original
controls, totals and output directory are NOT the active configuration.

## Previous design (stopped)

Run from `Codes`:

```powershell
python -u -m bsrt_communication_evaluation.run_token_ids
```

This run is resumable with the same command. It uses the unchanged full-reference
BSRT checkpoint and loads its existing compressed contexts. It does not retrain
BSRT or recompute any other compressor. Three seeds: 42, 123, 2024. The established
200-question/36-context channel cohort is retained. Targets: 10, 20, 30, 40%.

QPSK AWGN and TDL-A fading use SNR 0, 3, 5, 7, 10, 15 dB (high SNR runs first).
Packet-loss probabilities: 0, .01, .02, .05, .1, .2, .3. Each has uncoded and
Sionna 5G NR LDPC modes (512 information bits / 1024 code bits, 12 iterations).
Packet loss is a custom packet-erasure process, not a physical fading channel;
there is no additional noise on its surviving symbols.

Payloads are 15-bit BERT token IDs, not UTF-8 or embeddings. Original token-ID
controls are evaluated only once per context, channel, condition, seed and coding
mode, not once per compression level. Old UTF-8 and BPSK results cannot be reused
as equivalent controls. Only these required controls and BSRT are computed.
One received context is shared by all its associated questions.

Complete run: 91,200 BSRT outcomes + 22,800 original outcomes = 114,000 records.
Many outcomes share exact received contexts or clean cached predictions, so this
is not a count of fresh QA forward passes. Actual transport is simulated even
at high SNR; clean reception is checked, never assumed. Transport and QA cache
separately in `results_token_ids/transport_and_metrics.sqlite3`.

Dry run and independent smoke test:

```powershell
python -m bsrt_communication_evaluation.run_token_ids --dry-run
python -u -m bsrt_communication_evaluation.run_token_ids --smoke
python -m unittest bsrt_communication_evaluation.test_token_ids bsrt_communication_evaluation.test_channels
```

The smoke run uses a separate output directory and must never be presented as
the full experiment. Source/config/checkpoint/cohort hashes lock each output
directory; changing a locked protocol requires another directory.

Monitor the background runner:

```powershell
Get-Content ".\bsrt_communication_evaluation\results_token_ids\run.log" -Tail 15 -Wait
```

Completion marker: `Completed BSRT token-ID QPSK communication evaluation:`.
The report and summaries update throughout; final figures require every
condition. Generate figures again after completion:

```powershell
python -m bsrt_communication_evaluation.run_token_ids --plot
```

Figures use three channel rows and two coding columns; original controls have
the same coding mode as BSRT in each column. Accuracy replacement:
`results_token_ids/figures/final_communication_accuracy.pdf` (also SVG/PNG).
`channel_information_ber.pdf` checks that bit errors are actually simulated.

The PowerShell wrapper `run_bsrt_token_id_channels.ps1` runs the full sweep and,
only after successful completion, installs the accuracy figure into the thesis
with a backup. It refuses to overwrite a figure edited since the wrapper began.
Direct Python invocation never writes outside Codes.

Receiver assumptions and limitations are documented in the generated report:
known source lengths/packet framing, shared tokenizer, [UNK] for invalid IDs or
uncoded token erasures, ideal CSI for TDL, approximate equalized soft outputs,
equal symbol energy rather than equal total resources, and the historical QA
correctness rule. No claim of LDPC recovering every corrupted message is made.
