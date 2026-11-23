"""Write the latest report for the paired, non-singleton receiver subgroup."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results/full"
SUBSET = HERE / "results/non_singleton_receivers"
meta = json.loads((SUBSET / "subset_metadata.json").read_text(encoding="utf-8"))
lock = json.loads((SOURCE / "run_lock.json").read_text(encoding="utf-8"))
for filename, field in [("run_lock.json", "source_run_lock_sha256"),
                        ("receiver_results.csv", "source_receiver_results_sha256")]:
    assert hashlib.sha256((SOURCE / filename).read_bytes()).hexdigest() == meta[field]
summary = pd.read_csv(SUBSET / "summary_all_questions.csv")
clean = pd.read_csv(SUBSET / "clean_summary.csv").set_index("scenario")
cohort = pd.read_csv(SUBSET / "included_receivers.csv")
rows = pd.read_csv(SUBSET / "receiver_results.csv")
assert meta["questions"] == 416 and meta["receiver_outcomes"] == len(rows) == 29952
assert rows.groupby(["channel", "snr_db", "channel_seed", "scenario"]).size().eq(416).all()
actual = rows.groupby(["channel", "snr_db", "scenario"])[
    ["accuracy", "f1", "semantic_similarity", "information_ber", "decode_success"]].mean().sort_index()
np.testing.assert_allclose(actual, summary.set_index(["channel", "snr_db", "scenario"]).sort_index()[actual.columns], atol=1e-14)

groups = {g["context_id"]: g for g in lock["groups"]}
original_budget = sum(2 * ((15 * groups[c]["original_tokens"] + 1) // 2)
                      for c in cohort.context_id)
budget = meta["aggregate_budget_per_strategy_per_channel_snr_seed_bits"]
clusters = cohort.drop_duplicates(["context_id", "eligible_cluster_id"])
retained = sum(15 * len(groups[c]["individual_ids"][0]) for c in cohort.context_id)
shared = sum(15 * len(groups[c]["individual_ids"][0]) for c in clusters.context_id)


def qa_table(channel):
    lines = ["| Group-center SNR (dB) | Unicast accuracy (%) | Multicast accuracy (%) | Gain (percentage points) | Unicast F1 | Multicast F1 | Unicast similarity | Multicast similarity |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for snr in meta["snr_db_values"]:
        part = summary[summary.channel.eq(channel) & summary.snr_db.eq(snr)].set_index("scenario")
        u, m = part.loc["unicast"], part.loc["multicast"]
        lines.append(f"| {snr:g} | {100*u.accuracy:.2f} | {100*m.accuracy:.2f} | {100*(m.accuracy-u.accuracy):+.2f} | "
                     f"{u.f1:.4f} | {m.f1:.4f} | {u.semantic_similarity:.4f} | {m.semantic_similarity:.4f} |")
    return "\n".join(lines)


def ber_table(channel):
    lines = ["| Group-center SNR (dB) | Unicast mean information BER | Multicast mean information BER | Unicast error-free messages (%) | Multicast error-free messages (%) |",
             "|---:|---:|---:|---:|---:|"]
    for snr in meta["snr_db_values"]:
        part = summary[summary.channel.eq(channel) & summary.snr_db.eq(snr)].set_index("scenario")
        u, m = part.loc["unicast"], part.loc["multicast"]
        lines.append(f"| {snr:g} | {u.information_ber:.8g} | {m.information_ber:.8g} | {100*u.decode_success:.4f} | {100*m.decode_success:.4f} |")
    return "\n".join(lines)


clean_table = "\n".join(f"| {s.title()} | {100*clean.loc[s, 'accuracy']:.6f} | {clean.loc[s, 'f1']:.6f} | {clean.loc[s, 'semantic_similarity']:.6f} |"
                        for s in ["unicast", "multicast"])
histogram = "\n".join(f"| {int(size)} | {int(count)} | {int(size*count)} |"
                      for size, count in clusters.cluster_size.value_counts().sort_index().items())

text = rf"""# Latest LDPC experiment: unicast versus multicast, excluding singleton clusters

## Scope and revision

This report replaces the full-500 summary as the evidence for the revised plot.
It uses **416 receiver questions from 30 contexts in 109 task-alignment clusters**.
Every included cluster has at least two receivers. The 84 receivers belonging to
singleton clusters are excluded from **both unicast and multicast**. This is not
a filter on multicast alone and is not an exclusion based on observed QA success.

The underlying experiment evaluated 500 answerable SQuAD validation questions
from 36 contexts. Its original results remain unchanged. This report describes
a paired subgroup analysis of those saved results, not a new 500-question run.
There was no new compression, training, channel simulation or QA inference.

The original report is preserved as `results/full/FULL_COHORT_LDPC_EXPERIMENT_REPORT.md`.
Do not use its full-cohort numbers to describe the revised subgroup figure.

Research question: **Can task-aligned multicast reinvest avoided duplicate
transmissions in channel protection and improve receiver QA performance within
the same aggregate transmission budget?**

## 1. Comparison and subgroup selection

Both strategies use a nominal **40% token-removal target** and the same
**60% aggregate original-token-ID unicast budget**.

- **Unicast:** an individually task-compressed message is transmitted to each receiver.
- **Multicast:** one task-aligned compressed message is transmitted per cluster.

The plot legends are exactly **Unicast** and **Multicast**. The underlying methods
must still be documented: unicast has no FEC; multicast applies LDPC to the entire
retained payload and repeats coded bits where needed to use its assigned budget.
Protection is not restricted to selected important tokens within a retained frame.

Eligibility is determined from the original cluster membership in `run_lock.json`:
retain receiver q if and only if its original task-alignment cluster has size >=2.
This is cluster size, not the total number of questions for its context. Original
context groups already had more than one receiver, but could contain singleton clusters.
Whole eligible clusters are retained; no reclustering is performed.

| Cohort property | Value |
|---|---:|
| Original evaluated questions | 500 |
| Excluded singleton receivers/clusters | 84 |
| Included receiver questions | 416 |
| Included contexts | 30 |
| Included multicast clusters | 109 |
| Channel seeds | 3: 42, 123, 2024 |
| SNR targets per channel | 6: 0, 3, 5, 7, 10, 15 dB |
| Channels | AWGN and TDL multipath fading |
| Outcomes per curve point per strategy | 1,248 = 416 questions x 3 seeds |
| Total receiver outcomes in this report | 29,952 |

| Receivers per cluster | Included clusters | Receiver questions |
|---:|---:|---:|
{histogram}

Unicast and multicast contain exactly the same eligible questions at every
channel/SNR/seed condition. Their receiver SNRs, physical random seeds and
source-payload bit lengths are paired. Retained texts need not be identical.
The original full-context-group SNR offsets and seeds are unchanged: they are
not reassigned after removing singleton receivers. Questions with the same
context remain dependent; the number of seed outcomes is not a count of
independent contexts.

## 2. Importance, clustering and source compression

Token importance for receiver k is the ordinary input-embedding QA-loss gradient norm:

\[
p_{{k,t}}=\left\|\frac{{\partial L_{{\mathrm{{QA}},k}}}}{{\partial\mathbf e_t}}\right\|_2.
\]

Reference-answer spans define the supervised loss when available; the importance
helper falls back to negative predicted start/end logits if no usable span is found.
This is an **answer-supervised, task-aware reference experiment**, not the BSRT
retention policy, not question-blind compression, and not a newly trained model.

With equal 15-bit token costs, normalized receiver priorities are
\(\widetilde{{\mathbf v}}_k=(\mathbf p_k/15)/\|\mathbf p_k/15\|_2\).
The cosine kernel is \(\kappa(k,k')=\widetilde{{\mathbf v}}_k^\mathsf T\widetilde{{\mathbf v}}_{{k'}}\).
Average-linkage clustering uses distance \(\max(0,1-\kappa(k,k'))\).
Each cluster priority is the normalized sum of its members' unit vectors:

\[
\mathbf v_j^*=
\frac{{\sum_{{k\in\mathcal C_j}}\widetilde{{\mathbf v}}_k}}
{{\|\sum_{{k\in\mathcal C_j}}\widetilde{{\mathbf v}}_k\|_2}}.
\]

Original candidate counts are \(C=1,\ldots,\min(6,K_c)\). The selection objective is

\[
J(C)=\frac1{{K_c}}\sum_k
\left(1-\widetilde{{\mathbf v}}_k^\mathsf T\mathbf v_{{c(k)}}^*\right)
+0.08\frac{{C-1}}{{K_c-1}}.
\]

Candidates with minimum receiver-to-cluster alignment >=0.75 are preferred;
choose the minimum objective among feasible candidates, or among all candidates
if none is feasible. Ties favor fewer clusters. This is cosine misalignment,
not an arccos/angular loss. Clustering is independent of channel outcomes/SNR.

Unicast ranks each question's own importance; multicast ranks its cluster priority.
Both keep the highest-ranked allowed WordPieces and restore original order.
There is no whole-word completion rule. Integer framing means nominal 40% removal
need not mean precisely 40.000% for every context.

## 3. Equal budget after excluding singletons

For T_c original tokens, the original QPSK frame size, per-receiver budget and
retained token count are

\[
N_{{0,c}}=2\left\lceil\frac{{15T_c}}{{2}}\right\rceil,\qquad
b_c=2\left\lfloor\frac{{0.6N_{{0,c}}}}{{2}}\right\rfloor,\qquad
n_c=\left\lfloor\frac{{b_c}}{{15}}\right\rfloor.
\]

Let E_c be the eligible receiver set for context c. Unicast uses |E_c| frames
of b_c bits. Each retained multicast cluster of size m_j uses one frame of m_j b_c bits.
Since complete clusters are retained,

\[
B_{{\mathrm{{unicast}}}}=\sum_c |E_c|b_c
=\sum_{{j:\,m_j\ge2}}m_jb_{{c(j)}}=B_{{\mathrm{{multicast}}}}.
\]

The singleton budgets are removed from both sides, not redistributed to the
remaining receivers. All saved frames, code dimensions and repetition allocations
remain unchanged. LDPC mother-code rates are bounded below by 0.2; additional
capacity uses cyclic repetition with corresponding soft LLRs summed before
12-iteration decoding. Effective cluster information/wire rate is approximately
1/m_j, not 1/K_c in general. Source blocks support up to 254 retained tokens.

| Budget ledger: one channel/SNR/seed across the included cohort | Bits |
|---|---:|
| Original-token-ID unicast reference for the same 416 receivers | {original_budget:,} |
| Actual Unicast wire bits | {budget:,} |
| Actual Multicast wire bits, including protection | {budget:,} |
| Individual retained information bits, before padding | {retained:,} |
| Shared cluster information bits, before protection/padding | {shared:,} |

Both use **{100*budget/original_budget:.6f}%** of the same original-unicast reference,
or **{100*(1-budget/original_budget):.6f}% fewer bits**. There are 109 shared source
messages rather than 416 individual messages, but final wire costs are equal after
reinvesting savings. Equal aggregate symbols do not imply equal per-receiver energy,
latency or decoder cost. Neither plotted strategy transmits the original full context.

## 4. Channels, receiver and averaging

- Fixed-width 15-bit BERT WordPiece IDs, not UTF-8 text or embedding vectors.
- Unit-energy QPSK; SNR is E_s/N_0, not E_b/N_0.
- Group-center SNRs: 0, 3, 5, 7, 10 and 15 dB; seeds: 42, 123, 2024.
- Original context groups have ordered receiver offsets evenly spaced from -3
  to +3 dB. The horizontal coordinate is the group center, not every receiver's SNR.
- Sionna AWGN and normalized static SISO TDL multipath fading: delay spread
  30 ns, carrier 3.5 GHz, bandwidth 1 MHz, zero speed and sampled lags 0--4.
  The configured profile is A; the plot names it simply TDL multipath fading.
- Multipath reception uses ideal CSI and FFT-MMSE equalization, with approximate
  post-equalization noise/residual-ISI variance for LLRs; not a complete OFDM stack.
- A shared multicast waveform has separate noisy observations at its receivers.
- Received IDs are decoded into text before QA. QA checkpoint/tokenizer:
  `deepset/bert-base-cased-squad2`; max sequence 384, stride 128, max answer 15;
  no-answer handling disabled for this answerable cohort.
- Semantic similarity: `all-MiniLM-L6-v2`; maximum cosine similarity across references.
  Token-level F1 also uses the best reference.

Binary accuracy is the historical implemented criterion: normalized equality OR
normalized reference containment in prediction OR F1>=0.5 OR similarity>=0.75.
It is not official SQuAD Exact Match or manually verified factual correctness.
Threshold false positives remain possible; accompany accuracy with F1/similarity.

Each question has equal weight; first average its cohort within each seed, then
average the three seeds:

\[
\overline M(h,\gamma,s)=\frac13\sum_{{z\in\{{42,123,2024\}}}}
\frac1{{416}}\sum_{{q\in E}}M_{{q,z}}(h,\gamma,s).
\]

Do not describe these as equally weighted context means. Information BER is the
mean receiver-message error fraction after decoding, not pooled bit errors over
all transmitted bits. Error-free-message rate counts messages with zero residual
information errors; it is not QA accuracy. No significance tests or confidence
intervals have been calculated for this subgroup.

## 5. QA results for the revised plot cohort

### AWGN

{qa_table('awgn')}

### TDL multipath fading

{qa_table('fading')}

Gains are absolute percentage points, not relative percentages. Full-precision
values are in the subgroup `summary_all_questions.csv`.

## 6. Information-bit errors and message recovery

### AWGN

{ber_table('awgn')}

### TDL multipath fading

{ber_table('fading')}

Multicast has zero observed residual information BER from **5 dB in AWGN**
and **7 dB in multipath fading**. This is an observation for these messages/seeds,
not a guarantee for arbitrary channel realizations. At 5 dB fading, one of 1,248
multicast outcomes still has residual errors (99.9199% error-free messages).
Its QA accuracy is slightly above the clean value, so do not call that point
error-free or interpret its tiny non-monotonicity as a robust improvement.

## 7. Clean compressed-message reference

| Strategy | Accuracy (%) | F1 | Semantic similarity |
|---|---:|---:|---:|
{clean_table}

These are clean **compressed** messages on the same 416 questions, not original
uncompressed-context baselines. At high SNR, multicast reaches 90.384615% accuracy
and F1 0.830512; unicast reaches 91.586538% accuracy and F1 0.856363. Unicast's
clean advantage is **1.201923 percentage points** in accuracy and **0.025851** F1.
Sharing a priority across tasks changes the source representation, even before noise.

## 8. Interpretation and boundaries

1. Excluding singletons from both sides isolates receivers whose clusters actually
   provide sharing savings. It does not repair/change the full-system experiment.
2. At 3 dB AWGN, accuracy is 23.16% for Unicast and 87.74% for Multicast:
   **+64.58 percentage points**. At 5 dB fading it is 41.99% versus 90.46%:
   **+48.48 percentage points**. These are descriptive paired-cohort differences.
3. Avoided source-message copies fund redundancy within the same wire budget.
   This is a complete source-selection/grouping/coding comparison, not a coding-only ablation.
4. The old full-cohort gradual recovery partly reflected 84 singleton receivers.
   They are absent here; remaining multicast receivers reach clean performance
   earlier. Do not substitute these subgroup numbers as full-500 performance.
5. Multicast does not dominate at every SNR: Unicast is better at 15 dB because
   individually tailored clean messages retain a small source-quality advantage.
6. Group eligibility is determined by original clustering, not random assignment.
   This is conditional evidence for multi-receiver clusters, not a causal estimate
   generalizable to all receivers or every possible grouping.
7. Gold-answer/task supervision is used for importance. Do not identify this as
   a deployed question-blind compressor or BSRT policy.
8. Framing, cluster membership and coding dimensions are assumed known. Metadata,
   pilots, CRC, synchronization, scheduling, retransmissions and decoder energy
   are not budgeted. Equal aggregate wire costs are not equal latency guarantees.
9. Normalized channels and ideal CSI limit generalization to practical networks.
   Similar AWGN/TDL curves do not imply that fading is irrelevant in general.
10. No statistical significance is claimed; questions sharing contexts are dependent.

Suggested conclusion: **Among receivers in non-singleton task-alignment clusters,
multicast converts avoided duplicate transmissions into channel protection and
substantially improves QA robustness within the same aggregate token-ID budget.
At high SNR, individual unicast retains a small clean-message quality advantage.**

## 9. Figure, provenance and reproducibility

Revised figure (legends: **Unicast**, **Multicast** only):
`results/non_singleton_receivers/plots/unicast_vs_multicast_non_singleton_receivers.pdf`
with matching PNG/SVG and `_data.csv`. It shows accuracy and F1 for AWGN/TDL,
using straight segments through actual measured SNRs; there is no smoothing.
No existing LaTeX label is verified/assigned by this report.

Analysis files in `results/non_singleton_receivers`:

- `included_receivers.csv`: exact receiver IDs, context IDs, original cluster sizes and budgets.
- `receiver_results.csv`: 29,952 saved paired outcome records for the eligible cohort.
- `summary_per_seed.csv`, `summary_all_questions.csv`: seed-specific and three-seed means.
- `clean_summary.csv`: same-subset clean compressed-message means.
- `subset_metadata.json`: counts, source hashes, matching/budget checks and selection rule.

Authoritative unchanged source run: `results/full`, including `run_lock.json`,
`receiver_results.csv`, `clean_receiver_results.csv`, `budget_allocations.json`,
`status.json` and passed `audit.json`. The original full-cohort plot remains there.

Plot/subset code: `task_aligned_multicast_500/plot_non_singletons.py`.
Report code: `task_aligned_multicast_500/write_non_singleton_report.py`.
Underlying experiment: `run.py`, `cohort.py`, `transport.py`, `audit.py` in that folder;
relevance/clustering helper: `paper_multi_receiver_task_alignment/run_task_alignment.py`;
metrics: `bsrt_neural_compression/reward.py` and `main.py`.

Regenerate this report without simulation:
`python -m task_aligned_multicast_500.write_non_singleton_report`

## Instructions for thesis writing

Use **416 questions, 30 contexts, 109 clusters**, not 500 questions or 36 contexts,
when describing this figure. State that 84 singleton receivers were excluded from
both strategies using original cluster membership. Keep the simple legend labels,
but document the underlying source selection and LDPC/repetition in the methods.
Do not import numbers or p-values from earlier experiments. Do not describe the
baseline as uncompressed or hide the high-SNR reversal or task supervision.
"""

latest = SOURCE / "LATEST_LDPC_EXPERIMENT_REPORT.md"
archive = SOURCE / "FULL_COHORT_LDPC_EXPERIMENT_REPORT.md"
if not archive.exists():
    old = latest.read_text(encoding="utf-8")
    assert "## 1. Controlled comparison and evaluation cohort" in old
    shutil.copy2(latest, archive)
for path in [latest, SUBSET / "LATEST_LDPC_EXPERIMENT_REPORT.md"]:
    path.write_text(text, encoding="utf-8")
    print(path)
