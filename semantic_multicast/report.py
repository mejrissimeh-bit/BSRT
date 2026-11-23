"""Numerical report generated only from saved, independently decoded observations."""
from bsrt_neural_compression.common import resolve


def write_report(config, plan, status, summary, paired):
    lines = ["# Semantic multicast versus unicast", "",
        f"Status: {status['saved_cases']}/{plan['expected_cases']} cases; "
        f"{status['saved_receiver_outcomes']} receiver outcomes. Complete: {status['complete']}.", "",
        "## Reused components and evaluation cohort", "",
        f"{plan['contexts']} contexts, {plan['distinct_questions']} distinct QA tasks, "
        f"{plan['receiver_groups']} nested receiver groups. Groups per K: {plan['groups_by_K']}. "
        f"Unavailable groups: {plan['unavailable_groups_by_K']}. Channel seeds: {config['channel_seeds']}. "
        "Only final-cohort cached questions with verified context/question/reference identity are used. "
        "Contexts are selected in sorted hash order, capped by max_contexts. Within each context, "
        "a seeded permutation selects nested first-K question groups; outcomes never select groups. "
        "No training, checkpoint change or gradient regeneration occurs. groups.json records every task. "
        "Not every context supports every K; QA summaries use the available paired groups at each K. "
        "The resource-versus-K figure restricts to contexts supporting all displayed K values. "
        "Receiver groups overlap across K and conditions: they are NOT independent replicates.", "",
        "Receiver importance reuses the recorded norm of the QA span-loss gradient with respect "
        "to each input token embedding (run_task_alignment.py). Shared priority reuses its projected "
        "mean-angular-misalignment minimizer on normalized vectors. Since each transmitted ID now "
        "costs 15 bits, semantic density is relevance/15, NOT the earlier per-token UTF-8 cost. "
        "Unicast selects each receiver's highest-ranked tokens; multicast selects the shared highest-ranked "
        "tokens. Stable ties and original token order are preserved. floor(T*r/100) tokens are removed; "
        "the canonical cached context is truncated to at most 256 WordPieces, with no special tokens. "
        "This selection uses individual WordPieces, not BSRT whole-word groups. "
        "The existing estimator falls back to the negative predicted-span logits when no reference "
        "span is found, and to input-embedding norms for unmatched alignment positions. Those cached "
        "fallbacks are preserved, not silently recomputed. Cache files have estimator metadata but "
        "no checkpoint/source-code hash from their original creation; this run fingerprints the "
        "exact reused files, not a retrospectively proven generation environment.", "",
        "IMPORTANT: cached QA gradients depend on receiver questions AND reference-answer labels. "
        "These are task-aware supervised reference representations, not a question-blind deployable "
        "BSRT inference policy. Reference answers/importance vectors are not transmitted; their "
        "availability at the sender is an explicit experimental assumption. This tests communication "
        "of aligned task representations, not deployment without task knowledge.", "",
        "## Waveform and independent receivers", "",
        "Token IDs reuse ids_to_bits/received_ids/decode_ids. Each ID costs 15 MSB-first bits. "
        "Multicast creates ONE immutable encoded QPSK waveform, then feeds that identical array "
        "to separate Sionna channel calls and separate decoders. Each receiver has a distinct "
        "context/question/channel/profile/SNR/base-seed-hashed RNG seed. Seeds omit scenario and "
        "removal so comparable representations use paired random streams, without claiming identical "
        "finite noise arrays for different lengths. waveform_sha256 and physical_seed are saved. "
        "Unicast instead makes K individual frames. Even if content matches, those K transmissions "
        "are counted separately. Successful multicast decoding at one receiver never substitutes "
        "for another receiver's decoding.", "",
        f"Homogeneous receivers all use the configured center SNR. Heterogeneous SNRs are evenly "
        f"spaced across center +/- {config['heterogeneous_span_db']/2:g} dB in fixed receiver order. "
        "SNR means Es/N0 for unit-energy QPSK, not Eb/N0. AWGN and normalized SISO TDL fading "
        "reuse bsrt_communication_evaluation/channel.py unchanged. Fading has 30 ns delay spread, "
        "3.5 GHz carrier, 1 MHz bandwidth, static taps, sampled lags 0-4 and ideal-CSI FFT-MMSE "
        "equalization, with approximate residual-ISI/noise soft variance. No OFDM/pilot estimation "
        "or geometry-dependent path loss is added. Independent receiver channels are modeled, "
        "not spatially correlated fading.", "",
        "## Validated resource accounting", "",
        f"Fixed configured LDPC rate is {config['ldpc_rate']}; all receiver links use that same rate. "
        "There is NO link-adapted comparison. Sionna 5G NR rate-matched LDPC uses k=15*T_kept and "
        "n=2*ceil((k/R)/2), one block per frame, 12 decoder iterations. Uncoded n=2*ceil(k/2). "
        "The actual coded bits, modulation padding, and S=n/2 QPSK symbols are measured. "
        "Internal LDPC shortening/puncturing is handled by Sionna. Unicast aggregate information "
        "bits and symbols SUM all K frames; multicast counts its shared information and symbols "
        "ONCE, independent of receiver count. Resource savings = 1-S_multicast/S_unicast.", "",
        "Equal-per-representation comparison: both scenarios retain the same number of tokens "
        "per transmitted representation. Thus identical coded lengths imply exactly 1-1/K "
        "resource savings; this arithmetic is verified against actual encoded arrays, not a "
        "measured throughput gain. Equal-aggregate comparison: multicast retains its target "
        "token count, while unicast water-fills integer receiver token counts under the SAME "
        "aggregate coded-symbol ceiling. This necessarily compresses each unicast message more. "
        "Any integer-allocation remainder is recorded as unused_aggregate_bits, not hidden. "
        "These two comparisons must never be presented as equivalent.", "",
        "Payload length, code dimensions, tokenizer vocabulary, frame boundaries and task routing "
        "are ideal known side information. Headers, CRC, pilots, guard intervals, retransmissions, "
        "scheduling, encoder/decoder computation and feedback are not modeled or counted. "
        "TDL receive convolution tails are observations, not extra transmitted symbols. "
        "Consequently resources are modeled baseband symbols, not overhead-complete network costs. "
        "No measured spectral-efficiency, latency or throughput claim is supported.", "",
        "## Metrics", "",
        "decode_success means exact equality of every decoded information bit to the sent bits. "
        "It is an offline ground-truth diagnostic, NOT a CRC available to a real receiver. "
        "LDPC BLER is a one-codeword information-error indicator; uncoded frame_error is NOT LDPC BLER. "
        "BER divides information errors by information bits. Raw coded BER compares hard decisions "
        "before FEC with the actual transmitted codeword. Invalid/reserved IDs become [UNK]. "
        "all_receivers_decode is the measured conjunction across independent receiver outcomes, "
        "not the mean success rate raised to K.", "",
        "QA uses the existing deepset/bert-base-cased-squad2 pipeline (384 tokens, stride 128, "
        "max answer 15, no-answer mode off), token F1 and all-MiniLM-L6-v2 similarity. The historical "
        "binary criterion is normalized equality/reference substring OR F1>=0.5 OR similarity>=0.75; "
        "it is not manual factual correctness. Exact received-text/question/reference/settings metric "
        "keys reuse repeated QA results; the existing QA prediction cache is also reused. "
        "Group metrics first average K receivers, then average group/seed outcomes within each "
        "condition. receiver_summary.csv separately gives pooled receiver means. No confidence "
        "interval or significance claim is inferred from this small/overlapping group sweep.", "",
        "## Observed comparisons", "",
        "Every row below averages the saved contexts and channel seeds only; removal, K, coding, "
        "channel and SNR profile are kept distinct. Full SNR-level values are in summary.csv "
        "and paired_comparison.csv. Differences are multicast minus unicast.", "",
        "| Budget | Channel | Coding | Profile | K | Removal % | SNR dB | Savings % | Accuracy difference |",
        "|---|---|---|---|---:|---:|---:|---:|---:|"]
    columns = ["budget_regime", "channel", "coding", "snr_profile", "receiver_count", "removal_percent", "snr_db"]
    view = paired.groupby(columns)[["resource_savings", "accuracy_difference_multicast_minus_unicast"]].mean().reset_index()
    # Full numerical evidence lives in CSV; keep a small boundary-SNR report table.
    view = view[view.snr_db.isin([view.snr_db.min(), view.snr_db.max()]) & view.receiver_count.eq(view.receiver_count.max())]
    for row in view.itertuples(index=False):
        lines.append(f"| {row.budget_regime} | {row.channel} | {row.coding} | {row.snr_profile} | "
                     f"{row.receiver_count} | {row.removal_percent} | {row.snr_db:g} | "
                     f"{100*row.resource_savings:.2f} | {row.accuracy_difference_multicast_minus_unicast:+.4f} |")
    lines += ["", "Positive accuracy differences favor shared transmission; negative values expose lost "
              "task specialization or channel sensitivity. Under equal message budgets, fewer shared "
              "resources do not prove better task quality. Under equal aggregate budgets, savings "
              "should be approximately zero, and semantic quality is the relevant comparison. "
              "A weak receiver can prevent all-receiver success even when mean QA is high. "
              "Results are limited to this cached SQuAD subset and channel assumptions.", "",
              "## Artifacts and reproducibility", "",
              "run_lock.json fingerprints source implementations, gradients, groups and library versions; "
              "changed inputs require a new output folder. Each case commits receiver and group "
              "records atomically to SQLite, allowing safe resume after interruption. status.json "
              "tracks progress; numerical CSV exports and plots use committed observations only. "
              "Run commands and tests are in semantic_multicast/README.md."]
    (resolve(config["output_dir"])/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
