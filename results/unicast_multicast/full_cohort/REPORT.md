# Latest LDPC experiment: task-aligned multicast under a 60% aggregate budget

## Scope of this report

This report covers ONLY the completed experiment in `task_aligned_multicast_500/results/full`.
It does not merge earlier original-context multicast, K=2/3/5 global-alignment, BSRT
channel-coding, packet-loss or PCA results. No additional simulation or QA evaluation
was performed to write this report. Tables are computed from the recorded final data.

Suggested subsection subject: **Task-aligned multicast with budget-funded channel protection**.

Research question: Can sharing a compressed context among receivers with aligned
tasks release transmission resources for error correction, improving receiver QA
performance relative to individually compressed unicast under the same aggregate budget?

## 1. Controlled comparison and evaluation cohort

The source archive contains the final 500 SQuAD validation context--question pairs
already used in the thesis. There are **36 distinct contexts and 500 receiver questions**.
This is a reused answerable evaluation subset, not an evaluation of the full official
SQuAD benchmark. The report does not introduce a new random sample of 500 questions.
For context c, K_c is its actual question count: **2--29 receivers**. Its questions
are not subsampled to fixed K=2/3/5. All 500 questions contribute equally to final means.

The two scenarios are:

| Scenario | Source selection | Physical transmissions per context | Protection |
|---|---|---|---|
| Individual compressed unicast | Each question's own task importance selects its retained tokens | K_c separate messages | No FEC; unused sub-token capacity is padding |
| Task-aligned compressed multicast | One shared retained-token sequence per task-alignment cluster | C_c cluster messages | LDPC for clusters with at least two receivers; repetition when required; singleton clusters remain uncoded |

Both scenarios use a nominal **40% token-removal target**, implemented under a
strict **60% original-unicast token-ID bit budget**. The baseline in this experiment
is therefore **compressed individual unicast, NOT uncompressed original-context unicast**.
These are different retained texts in general; the source-selection rules and
channel-coding arrangements are compared as complete systems.

The full original contexts are tokenized without truncation or special tokens.
Original lengths range from **35 to 418 WordPieces**. Four older relevance caches
stopped at 256 tokens; only missing full-length gradients were extended and cached,
with previously compatible extended gradients reused. Hidden truncation is not
counted as token compression. Integer token/QPSK framing gives achieved token
removal between **40.000000% and 42.857143%**,
with a question-weighted mean of **40.563572%**. Thus 40% is a
target, not an assertion that every short context loses exactly 40.000% of its tokens.

## 2. Importance, task alignment and token selection

The importance of token t for receiver k is the L2 norm of the QA-loss gradient
with respect to that token's input embedding:

\[
p_{k,t}=\left\|\frac{\partial L_{\mathrm{QA},k}}{\partial\mathbf e_t}\right\|_2.
\]

Reference-answer spans supply the supervised QA objective when a span is found.
The reused importance function falls back to the negative predicted start/end
logits when no usable reference span is found. This is an **answer-supervised,
task-aware reference experiment**, not a question-blind deployed compressor.
It uses questions and reference-answer information for the evaluated cohort's
importance estimates. It must not be described as a learned BSRT policy or as
unseen-task, question-independent compression. No new retention model is trained.
The A0 configuration is reused for the receiver QA checkpoint/settings only;
the BSRT retention network is not used to select these messages.

Every token ID has the same 15-bit cost. Normalize each receiver vector after
division by that cost:

\[
\widetilde{\mathbf v}_k=
\frac{\mathbf p_k/15}{\|\mathbf p_k/15\|_2},\qquad
\kappa(k,k')=\widetilde{\mathbf v}_k^\mathsf{T}
\widetilde{\mathbf v}_{k'}.
\]

Average-linkage agglomerative clustering uses distance
\(\max(0,1-\kappa(k,k'))\). Candidate cluster counts are
\(C=1,\ldots,\min(6,K_c)\). For cluster j, the shared priority is the
normalized sum of its receivers' unit vectors:

\[
\mathbf v_j^*=
\frac{\sum_{k\in\mathcal C_j}\widetilde{\mathbf v}_k}
{\left\|\sum_{k\in\mathcal C_j}\widetilde{\mathbf v}_k\right\|_2}.
\]

Cluster selection minimizes

\[
J(C)=\frac{1}{K_c}\sum_{k=1}^{K_c}
\left(1-\widetilde{\mathbf v}_k^\mathsf{T}\mathbf v_{c(k)}^*\right)
+0.08\frac{C-1}{K_c-1}.
\]

The implementation first considers candidates with minimum receiver-to-cluster
alignment at least **0.75**. It chooses the smallest objective among feasible
candidates, or among all candidates if none is feasible; ties favor fewer clusters.
Clustering does not depend on SNR or observed post-channel QA outcomes.
The loss here is **cosine misalignment**, not mean arccos/angular distance.

Individual unicast selects the highest-ranked tokens from each receiver's own
importance vector. Multicast selects from each cluster's shared priority. The
retained tokens are restored to original order. Selection is at WordPiece level;
there is no BSRT-style whole-word completion rule in this experiment.

## 3. Observed clusters and available protection

There are **193 clusters in total**, versus 500 individual receiver representations:
**61.4% fewer source-message transmissions before reinvesting sharing savings**.
This is not a 61.4% reduction in final transmitted bits: both final systems spend
the same aggregate bit budget after coding/padding.

Cluster counts per context range from 2 to 6. Across all contexts:

| Receivers per cluster | Number of clusters | Receiver questions covered |
|---:|---:|---:|
| 1 | 84 | 84 |
| 2 | 39 | 78 |
| 3 | 20 | 60 |
| 4 | 17 | 68 |
| 5 | 14 | 70 |
| 6 | 9 | 54 |
| 7 | 3 | 21 |
| 8 | 2 | 16 |
| 9 | 2 | 18 |
| 10 | 2 | 20 |
| 11 | 1 | 11 |

**84 singleton clusters cover 84 questions (16.8%)**. They have no avoided duplicate
transmissions and remain uncoded. The other **109 clusters cover 416 questions
(83.2%)** and receive LDPC protection. **32 clusters**
also require explicit repetition to spend their assigned budgets below the
supported LDPC mother-code rate. Maximum observed cluster size is 11.

## 4. Exact token-ID budget and channel coding

Text is represented by fixed-width **15-bit BERT WordPiece token IDs**, not raw text
bytes or embedding vectors. For T_c original tokens, define the uncoded original
QPSK-frame length

\[
N_{0,c}=2\left\lceil\frac{15T_c}{2}\right\rceil.
\]

Each receiver's allowed budget and retained-token count are

\[
b_c=2\left\lfloor\frac{0.6N_{0,c}}{2}\right\rfloor,
\qquad n_c=\left\lfloor\frac{b_c}{15}\right\rfloor.
\]

The context-wide budget is

\[
B_c=K_cb_c\le 0.6K_cN_{0,c}.
\]

Unicast sends K_c distinct frames of b_c bits each. The payload occupies
15n_c bits, with any remaining sub-token capacity filled by zeros, not FEC.
The absence of coding is a design choice at this retained-token budget, not a
claim that unicast can never be coded under other source/rate choices.

For a multicast cluster of m_j receivers, its one shared frame is allocated
\(N_{c,j}=m_jb_c\) bits. Therefore

\[
\sum_{j=1}^{C_c}N_{c,j}
=\sum_{j=1}^{C_c}m_jb_c=K_cb_c=B_c.
\]

The effective information/wire rate for that cluster is

\[
R_{c,j}=\frac{15n_c}{m_jb_c}\approx\frac{1}{m_j}.
\]

It is **not generally 1/K_c**, because each context can have multiple clusters.
For m_j=1 there is no coding; for m_j>=2, Sionna 5G LDPC encodes the retained
message with 12 soft-decoding iterations. The supported mother-code rate is
bounded below by 0.2. When the assigned wire length exceeds the supported
mother codeword, coded bits are repeated cyclically. Corresponding received soft
LLRs are **summed before LDPC decoding**. No source token is duplicated in the
receiver's reconstructed text. Thus this is **LDPC plus repetition where needed**,
not a native arbitrarily low-rate LDPC code. The code supports segmentation at
254 source tokens; it retains every selected source token.

Illustrative budget example (not an extra experimental result): 100 original
tokens cost 1,500 original bits. Each receiver gets b=900 bits and n=60 retained
tokens. Ten receivers have B=9,000 bits. With cluster sizes 4,3,2,1, multicast
sends four frames of 3,600, 2,700, 1,800 and 900 bits; unicast sends ten 900-bit
frames. The singleton is uncoded and aggregate costs are equal.

Actual bit ledger for one complete evaluation of the 500 questions at one
channel/SNR/seed condition (36 contexts served once):

| Quantity | Bits |
|---|---:|
| Full-context uncoded unicast reference, including QPSK padding | 1,063,304 |
| Actual individual compressed unicast transmission | 637,648 |
| Actual task-aligned multicast transmission, including coding/repetition | 637,648 |
| Compressed individual information payload, excluding padding | 633,435 |
| Cluster information payload before coding/padding | 248,535 |

Both actual systems use **59.968551%** of the original-unicast
reference bits, or **40.031449% fewer bits**. Equal wire
length also gives equal aggregate unit-energy QPSK-symbol budgets. It does not
equalize per-receiver received energy, latency or computational cost.

## 5. Physical channel and QA receiver

- QPSK with unit symbol energy; SNR means **E_s/N_0**, not E_b/N_0.
- Group-center SNRs: **0, 3, 5, 7, 10 and 15 dB**; not a 0.5 dB sweep.
- Channel seeds: **42, 123 and 2024**.
- Each full context's K_c receivers have deterministic ordered SNR offsets
  evenly spaced from -3 to +3 dB around the group center. These offsets represent
  heterogeneous link qualities; the plotted SNR is not every receiver's SNR.
- The same receiver's actual SNR and physical seed match across both scenarios.
  Receivers have separate channel observations. A cluster shares one transmitted
  waveform, not one common received/noisy text.
- Sionna AWGN and the reused normalized, static SISO TDL multipath channel are
  simulated. The configured TDL profile is A; thesis-facing titles can simply
  say **TDL multipath fading**. Delay spread is 30 ns, carrier frequency 3.5 GHz,
  bandwidth 1 MHz, zero receiver speed, and discrete time-channel lags 0--4.
  These five sampled lags must not be mislabeled as the physical TDL path count.
- The multipath receiver uses ideal channel knowledge and FFT-domain MMSE
  equalization; the LLR variance accounts approximately for post-equalization
  noise and residual ISI. This is not a complete OFDM/pilot/synchronization stack.
- Bits are demodulated, repeated coded observations combined if present, and
  LDPC-decoded. The receiver reconstructs token IDs and decodes them into text
  before running QA. Framing and code dimensions are assumed known.
- QA checkpoint/tokenizer: **deepset/bert-base-cased-squad2**; inference uses
  max sequence length 384, stride 128 and maximum answer length 15. No-answer
  handling is disabled for this answerable cohort. The QA receiver is not retrained.
- Semantic similarity: **all-MiniLM-L6-v2**, maximum cosine similarity against
  the saved reference answers. Token-level F1 likewise takes the best reference.
- Locked software: torch 2.13.0, sionna 2.0.1, numpy 2.4.4, transformers 4.57.6, scikit-learn 1.8.0.

## 6. Metrics and averaging

Binary accuracy is the historical implemented correctness indicator: normalized
prediction/reference equality, OR containment of a normalized reference in the
prediction, OR token-level F1>=0.5, OR semantic similarity>=0.75. It is **not official
SQuAD Exact Match and not a manual factual-correctness audit**. In particular,
threshold-based false positives remain possible. Do not state an AND criterion
or substitute a 0.75 F1 threshold. This report focuses on accuracy, F1 and semantic
similarity, not Exact Match.

For each channel/SNR/seed/scenario, average over all 500 question receivers;
then average the three seeds. Equivalently for the balanced final run:

\[
\overline M(h,\gamma,s)=\frac{1}{3}
\sum_{z\in\{42,123,2024\}}
\frac{1}{500}\sum_{q=1}^{500}M_{q,z}(h,\gamma,s).
\]

Questions are not first averaged into 36 equally weighted context values, nor
are fixed K groups averaged equally. Each plotted condition has **1,500 receiver
outcomes** per strategy (500 questions x 3 seeds).

Information BER is the **mean of each receiver message's error fraction after
decoding**, not a pooled total-error/total-bit ratio. For uncoded frames this
is measured after hard decisions. Decode success is the fraction of receiver
messages with zero information-bit errors, not the fraction of correct QA answers.
No confidence intervals, hypothesis tests or significance claims have been
calculated for this latest experiment. Existing p-values from other experiments
must not be imported into this section.

## 7. Final QA results

### AWGN

| Group-center SNR (dB) | Unicast accuracy (%) | Multicast accuracy (%) | Gain (percentage points) | Unicast F1 | Multicast F1 | Unicast similarity | Multicast similarity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.60 | 62.53 | +54.93 | 0.0453 | 0.5689 | 0.2949 | 0.7107 |
| 3 | 23.20 | 76.93 | +53.73 | 0.1871 | 0.7040 | 0.4338 | 0.8190 |
| 5 | 44.80 | 82.40 | +37.60 | 0.3758 | 0.7517 | 0.5912 | 0.8611 |
| 7 | 65.93 | 85.27 | +19.33 | 0.5762 | 0.7773 | 0.7449 | 0.8844 |
| 10 | 87.93 | 90.00 | +2.07 | 0.7952 | 0.8191 | 0.8967 | 0.9184 |
| 15 | 91.40 | 90.40 | -1.00 | 0.8476 | 0.8261 | 0.9353 | 0.9243 |

### TDL multipath fading

| Group-center SNR (dB) | Unicast accuracy (%) | Multicast accuracy (%) | Gain (percentage points) | Unicast F1 | Multicast F1 | Unicast similarity | Multicast similarity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.87 | 63.13 | +56.27 | 0.0412 | 0.5736 | 0.2948 | 0.7158 |
| 3 | 22.93 | 76.67 | +53.73 | 0.1813 | 0.6985 | 0.4308 | 0.8166 |
| 5 | 41.93 | 82.27 | +40.33 | 0.3538 | 0.7441 | 0.5746 | 0.8573 |
| 7 | 63.40 | 85.07 | +21.67 | 0.5649 | 0.7772 | 0.7305 | 0.8840 |
| 10 | 86.47 | 89.73 | +3.27 | 0.7872 | 0.8156 | 0.8924 | 0.9161 |
| 15 | 91.40 | 90.40 | -1.00 | 0.8476 | 0.8261 | 0.9353 | 0.9243 |

All gains are **absolute percentage-point differences**, not relative percentage
improvements. Tables are descriptive means from the completed paired run.
Full-precision values remain in `summary_all_questions.csv`.

## 8. Information-bit errors and complete-message recovery

### AWGN

| Group-center SNR (dB) | Unicast mean information BER | Multicast mean information BER | Unicast error-free message (%) | Multicast error-free message (%) |
|---:|---:|---:|---:|---:|
| 0 | 0.158613 | 0.0550884 | 0.00 | 65.53 |
| 3 | 0.0834186 | 0.0176841 | 0.00 | 79.53 |
| 5 | 0.0444512 | 0.00798026 | 0.13 | 83.20 |
| 7 | 0.018911 | 0.00353969 | 7.00 | 84.20 |
| 10 | 0.00295384 | 0.000666596 | 46.27 | 91.00 |
| 15 | 7.14932e-06 | 1.48904e-06 | 99.13 | 99.73 |

### TDL multipath fading

| Group-center SNR (dB) | Unicast mean information BER | Multicast mean information BER | Unicast error-free message (%) | Multicast error-free message (%) |
|---:|---:|---:|---:|---:|
| 0 | 0.159239 | 0.0558499 | 0.00 | 65.33 |
| 3 | 0.0844486 | 0.0176211 | 0.00 | 79.27 |
| 5 | 0.045444 | 0.00822543 | 0.00 | 83.13 |
| 7 | 0.0193189 | 0.00358865 | 6.60 | 84.33 |
| 10 | 0.00301666 | 0.00064284 | 45.40 | 90.93 |
| 15 | 1.31487e-05 | 2.04015e-06 | 98.80 | 99.60 |

At 15 dB, mean information BER is very small but **not identically zero** in
the overall system. Nevertheless the recorded QA means equal the corresponding
clean-message QA means. A zero QA degradation does not imply every bit is correct.

## 9. Clean compressed-message reference and the singleton effect

These clean results evaluate each scenario's selected compressed text before
channel corruption. They are **not uncompressed-context baselines**.

| Scenario | Accuracy (%) | F1 | Semantic similarity |
|---|---:|---:|---:|
| Individual compressed unicast | 91.40 | 0.847615 | 0.935336 |
| Task-aligned compressed multicast | 90.40 | 0.826107 | 0.924330 |

Sharing task-specific information costs approximately **1.00 percentage point**
of clean accuracy here: 90.4% versus 91.4%. Clean F1 is also lower by
0.021508. Accordingly, after channel
impairments become sufficiently small, individually tailored unicast can regain
its small source-quality advantage. This is visible at 15 dB in both channels.

The current multicast system does **not** saturate completely at 3 dB. Its overall
AWGN accuracy rises from 76.93% at 3 dB to 90.40% at 15 dB. A major reason is that
84 singleton receivers remain uncoded. The saved coding-subgroup results show:

- At 5 dB AWGN, all **416 LDPC-protected receivers**, across all three seeds,
  have zero residual information BER and 100% complete-message decoding success.
  Their QA accuracy is **90.384615%**.
- The **84 uncoded singleton receivers** at that same condition have mean BER
  **0.04750156**, complete-message success
  **0.00%**, and QA accuracy
  **42.857143%**.
- Weighting those groups by their question counts yields the observed overall
  multicast accuracy **82.40%**, not the protected subgroup's approximately 90.38%.

`REPORT_coding_subgroups.csv` supplies the equivalent breakdown at every SNR in
both channels. Subgroup differences are descriptive, not randomized causal
comparisons: different question subsets belong to the two groups.

## 10. Supported interpretation for thesis writing

1. At the same 60% aggregate original-unicast budget, task-aligned compressed
   multicast achieves much higher receiver QA performance than individually
   compressed uncoded unicast at low and moderate SNR.
2. For example, at 3 dB AWGN, accuracy rises from 23.20% to 76.93%
   (**+53.73 percentage points**), and F1 from 0.1871 to 0.7040. At 5 dB TDL
   fading, accuracy rises from 41.93% to 82.27% (**+40.33 percentage points**).
3. The mechanism is resource reallocation: receivers in a cluster share one
   source message, and avoided copies fund coding/repetition. Redundancy does
   not exceed the aggregate budget, but it is not free in physical resources.
4. Compression, task-aware grouping and coding are jointly evaluated. This is
   not a coding-only ablation, not proof that semantic clustering is globally
   optimal, and not a comparison with ordinary identical-full-context multicast.
5. Gains are not universal across SNR. At 15 dB, individual unicast reaches
   91.40% versus 90.40% for multicast, reflecting its better clean task-specific
   source representation. It would be incorrect to claim multicast dominates
   unicast at every SNR.
6. The protected subgroup can reach clean-message performance before the whole
   multicast system does, because singleton clusters cannot fund extra protection.
7. State the outcome as **task-aligned multicast enables budget-funded channel
   protection under the modeled assumptions**, not as a universal robustness
   guarantee or as question-blind semantic compression.

Suggested concise conclusion: Task-aligned multicast reallocates avoided
duplicate transmissions to channel protection and substantially improves QA
performance in noisy conditions within the same aggregate token-ID budget.
At high SNR, the benefit diminishes and individual task-specific compression
retains a small clean-quality advantage.

## 11. Boundaries and limitations that must remain explicit

- Gold-answer/task supervision is used to derive the evaluation cohort's token
  importance. This is not an independently deployed question-blind policy.
- Both systems are compressed. Do not describe this latest unicast curve as
  transmission of the original uncompressed context.
- The budget reference is full original **token-ID unicast**, not original text
  bytes, one multicast copy, PCA coefficients or an earlier experiment's budget.
- Counting transmission copies makes multicast cheaper at the source-message
  stage; reinvesting those savings makes final aggregate transmitted bits equal.
- Singleton clusters receive no coding. Protection is not applied only to the
  most important tokens within a frame: the entire retained cluster payload is
  LDPC-coded when the cluster has at least two members.
- Some clusters use LDPC plus repeated coded bits, not LDPC alone at rate below 0.2.
- Cluster/frame ordering, membership, source lengths, token-ID boundaries and
  coding parameters are assumed known. Metadata, pilots, CRC, cluster signaling,
  synchronization, retransmissions, scheduling, finite airtime competition and
  decoder energy are excluded from the budget.
- The channel is a controlled normalized baseband model with ideal CSI, not a
  measured network or a distance/path-loss simulation. Similar AWGN/TDL curves
  are not evidence that all real fading channels behave like AWGN.
- The context groups share questions and contexts, so dependence must be respected
  in any future inference. Three channel seeds are not 1,500 independent contexts.
- No statistical significance has been established for this latest experiment.
- Historical binary-correctness thresholds may over-accept some answers; F1 and
  semantic similarity should accompany accuracy, as in the result tables.

## 12. Completion, verification, cache reuse and exact sources

Completed: **2,592 context-condition-strategy cases**, **36,000 receiver outcomes**,
and **1,296 paired context/channel/SNR/seed comparisons**. There are 72 equally
sampled strategy/channel/SNR/seed combinations, each with all 500 receivers.
The source audit checks identical receiver identities, actual SNRs and physical
seeds, exact aggregate budget equality, within-cluster common waveforms, bit-error
denominators and clean-message consistency whenever residual information errors
are zero. It passed. A final audit initially used a floating-point 0.6 comparison;
it was corrected to exact integer arithmetic. There were no budget violations,
and no transmission was rerun for that audit repair.

Cached gradients, compatible earlier predictions and exact matching transport
records were eligible for reuse. The audited end-to-end pilot supplied eight
cases, 50 clean results, 157 metric entries and 196 transport entries. Run-local
transport hits avoid repeating matching received waveforms. No nominal-SNR means
from an older, different experiment are substituted as the current baseline.
The final run's counters are computational/cache bookkeeping, not sample counts.

Authoritative public experiment directory:
`results/unicast_multicast/full_cohort`

Main figure (AWGN/TDL accuracy and F1, question-weighted):
`plots/task_aligned_multicast_vs_individual_unicast_40_percent.pdf`
with matching `.png` and `.svg`. No LaTeX figure label is assigned by this report.
Suggested new label if inserted into the thesis:
`fig:task_aligned_multicast_budgeted_fec` (a proposal, not an existing verified label).

Primary files:

- `summary_all_questions.csv`: reported final metric means.
- `summary_per_seed.csv`: each seed's question-weighted mean.
- `receiver_results.csv`: paired question-level outcomes, received text and BER.
- `context_results.csv`: group metadata and exact per-context transmission budgets.
- `clean_summary.csv`, `clean_receiver_results.csv`: clean compressed-message baselines.
- `budget_allocations.json`: cluster memberships, code dimensions, repeated bits and costs.
- `run_lock.json`: actual configuration, selected IDs, questions, importance provenance,
  implementation hashes and software versions. Prefer it over a subsequently edited config.
- `audit.json`, `status.json`: passed audit and full completion.
- `REPORT_coding_subgroups.csv`: derived protected/singleton breakdown for this report.

Code (relative to Codes): `task_aligned_multicast_500/run.py`, `cohort.py`,
`transport.py`, `audit.py`, `plot.py`; reused task relevance/clustering:
`paper_multi_receiver_task_alignment/run_task_alignment.py`; reused channel and
decoder helpers: `bsrt_communication_evaluation/channel.py`,
`multicast_coding_budget/transport.py`, `multicast_coding_budget/compressed_transport.py`,
`bsrt_budgeted_ldpc/channel.py`; QA metrics: `bsrt_neural_compression/reward.py`
and `main.py`. These code references do not imply importing other experiments' results.

## Instructions for a thesis-writing assistant

Use only this report's latest 500-question experiment and linked raw results.
Present the controlled source/coding/budget setup before its results. Retain the
distinction between K receivers and C clusters, the conditional LDPC/repetition
mechanism, and the singleton limitation. Describe the low-SNR gains and the
high-SNR reversal honestly. Do not add p-values, claim universal superiority,
call the baseline uncompressed, identify the compressor as BSRT, or characterize
the task-gradient supervision as question-blind. Do not merge the earlier
3 dB-saturating global multicast experiment or the earlier BSRT LDPC experiment.
