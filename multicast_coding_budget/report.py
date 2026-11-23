from bsrt_neural_compression.common import resolve


def write_report(config, plan, status, summary):
    lines = ["# Multicast savings fund channel coding: identical original tokens", "",
        f"Status: {status['saved_cases']}/{plan['expected_cases']} cases; complete={status['complete']}. "
        f"Receiver outcomes: {status['saved_receiver_outcomes']}/{plan['expected_receiver_outcomes']}.", "",
        "## Controlled comparison", "",
        "Unicast sends the full original token sequence separately to every receiver, without FEC. "
        "Multicast sends the SAME full token sequence once, protected with Sionna LDPC. There is no "
        "token removal, semantic alignment, importance estimation or retraining in this experiment. "
        "The sender needs the context and group size, not receiver questions or answer labels. "
        "This isolates multicast's opportunity to redirect avoided duplicate transmissions into FEC. "
        "It is not evidence for a semantic-compression algorithm by itself.", "",
        f"The saved cohort supplies {plan['contexts']} common contexts, {plan['groups']} nested K=2/3/5 "
        f"groups and {plan['distinct_questions']} distinct questions. Original texts are loaded from "
        "the archived evaluation CSV and tokenized WITHOUT truncation or special tokens using "
        "deepset/bert-base-cased-squad2. Unlike the prior alignment experiment, the payload is not "
        f"clipped to 256 tokens. Original lengths: {plan['source_tokens_min']}--{plan['source_tokens_max']} "
        "WordPieces. Questions and context hashes are checked against the saved cohort. The optional "
        "10% fallback was NOT needed or used; both scenarios transmit identical original IDs.", "",
        "## Exact aggregate budget", "",
        "For T original tokens, L=15T information bits and N0=2*ceil(L/2) uncoded QPSK bits. "
        "Unicast costs B=K*N0 bits, or B/2 symbols, because it sends K complete messages. "
        "Multicast has ONE shared waveform and spends at most B bits on LDPC. Its effective "
        "rate is approximately L/B=1/K: about 1/2, 1/3 and 1/5 for K=2,3,5. Thus improvements "
        "are measured at no greater AGGREGATE modeled symbols/energy, but multicast frames last "
        "longer than an individual unicast frame. Per-receiver energy, latency and scheduling "
        "are not held equal or measured. There is no claim that LDPC redundancy is free by itself.", "",
        "Sionna's auto-selected low-rate BG2 requires short information blocks; BG1 does not support "
        "rates below 1/3, and no 5G block supports rates below 1/5 here. Original payloads are "
        "therefore split into balanced contiguous blocks of at most 254 tokens (3810 information "
        "bits, below the 3824 auto-BG2 selection threshold). Every original token is included once. "
        "Each block receives a rate-matched even coded length. At K=5 the 1/5 bound and QPSK "
        "integer rounding may leave a few aggregate budget bits unused; these are recorded, "
        "never exceeded. Internal LDPC shortening/puncturing is handled by Sionna; no source "
        "padding tokens are introduced. Twelve soft decoding iterations are used.", "",
        "## Channel, receiver and caching", "",
        f"QPSK, SNR centers {config['snr_db_values']} dB, seeds {config['channel_seeds']}. "
        f"Heterogeneous receiver SNRs span center +/- {config['heterogeneous_span_db']/2:g} dB. "
        "Independent receiver seeds match between the two scenarios. The same unit-energy QPSK "
        "AWGN/TDL and ideal-CSI FFT-MMSE receiver as previous work is reused. SNR is Es/N0. "
        "A common immutable multicast waveform is prepared once per group and separately decoded "
        "at each receiver. Encoded blocks are concatenated into that one waveform. Unicast "
        "transmits the original frame K times and counts all K copies, even though their content matches.", "",
        "The previous uncompressed archive has different context-only physical seeds, some different "
        "cohort questions and only uncoded original results; its means are NOT substituted as a paired "
        "baseline. Missing receiver transmissions are simulated. Existing exact received-text/question/"
        "reference/model/metric matches are reused read-only from previous metric caches, with the "
        "proper historical key schema. The shared QA prediction cache is reused too. No nominal-SNR "
        "or clean-text assumption permits reusing a different received signal. Completed new cases "
        "and exact new transport entries support resume.", "",
        "QA: existing deepset BERT pipeline, max length 384, stride 128, maximum answer length 15, "
        "no-answer mode off. F1 and all-MiniLM-L6-v2 similarity are unchanged. Binary accuracy uses "
        "normalized equality/reference substring OR F1>=0.5 OR similarity>=0.75, not exact match "
        "or an independent factual audit. BER is measured after decoding; all-receiver success is "
        "the actual conjunction, not inferred from one receiver. LDPC BLER averages erroneous "
        "information blocks; frame error means any error in the reconstructed full message. "
        "Code dimensions, lengths, tokenizer and frame boundaries are ideal known side information. "
        "Headers, CRC, pilots, retransmissions, block-boundary signaling, latency and decoder energy "
        "are not modeled. These are aggregate baseband-symbol budgets, not full network costs.", "",
        "## Results", "",
        "Each point first averages receivers, then contexts and seeds within K, then weights "
        "K=2,3,5 equally. Partial-run means may lack K values: do not treat them as final. "
        "No significance test or confidence interval is claimed. The 3 dB gain is a hypothesis "
        "to measure, not a hard-coded expected outcome; weak receivers can still fail.", "",
        "| Channel | Center SNR dB | Scenario | Accuracy | F1 | Similarity | Information BER | All decode |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples(index=False):
        lines.append(f"| {r.channel} | {r.snr_db:g} | {r.scenario} | {r.mean_accuracy:.4f} | {r.mean_f1:.4f} | "
                     f"{r.mean_semantic_similarity:.4f} | {r.mean_information_ber:.6f} | {r.all_receivers_decode:.4f} |")
    lines += ["", "Full per-K scores and resource accounting: summary_per_k.csv and budget_allocations.json. "
              "Every receiver's question, references, received text, prediction, waveform hash, SNR "
              "and bit-error diagnostics are retained in receiver_results.csv. run_lock.json freezes "
              "the full payload identity and implementation. No previous thesis figure is overwritten."]
    (resolve(config["output_dir"])/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
