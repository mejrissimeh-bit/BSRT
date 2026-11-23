# Paper-Style Multi-Receiver Task Alignment

This experiment instantiates the multi-receiver formulation from
"Measuring Meaning: A Hilbert Space Approach to Semantic Transmissions"
for the thesis QA setting.

For every context, each receiver task induces a token-level semantic density
vector. The prioritization operator is instantiated with the L2 norm of the QA
distortion gradient for each context token. Receiver alignment is measured with
a cosine kernel. Similar receivers are clustered and one shared compressed
context is produced per cluster.

## Run

Inspect the dataset and estimated cost without loading the models:

```powershell
python ".\paper_multi_receiver_task_alignment\run_task_alignment.py" --dry-run
```

Run a small end-to-end check:

```powershell
python ".\paper_multi_receiver_task_alignment\run_task_alignment.py" --max-contexts 1
```

Run the complete experiment:

```powershell
python ".\paper_multi_receiver_task_alignment\run_task_alignment.py"
```

### Explicit angular-misalignment Global SA

This variant keeps the same receiver importance vectors, clustering, budgets,
and evaluation, but obtains the single Global SA co-priority by numerically
minimizing the mean angular distance from paper Eq. (9). It reuses the existing
relevance cache and writes to a separate result directory:

```powershell
python ".\paper_multi_receiver_task_alignment\run_task_alignment.py" --config ".\paper_multi_receiver_task_alignment\config_angular_global_sa.json"
```

Regenerate plots and reports from completed results:

```powershell
python ".\paper_multi_receiver_task_alignment\run_task_alignment.py" --plot-only
```

The run is resumable. Per-context relevance vectors are cached and completed
receiver evaluations are skipped.

Two accuracy aggregations are produced. `mean_accuracy.svg` gives every
transmitted context equal weight. `mean_accuracy_all_questions.svg` gives every
receiver question equal weight, so contexts with more receiver questions have
more influence.

## Paper Equations Used

- Eq. (3)-(4): the otherwise abstract prioritization operator is instantiated
  using the L2 norm of a task-specific QA distortion gradient, one of the
  estimator families explicitly allowed by the paper.
- Eq. (5)-(6): semantic density is relevance divided by token transmission cost.
- Eq. (8), (20)-(21): cosine alignment produces the receiver kernel matrix.
- Eq. (9): a cluster co-prioritization minimizes average angular misalignment.
- Sec. V-B: aligned receivers share one representation; dissimilar groups get
  separate representations.

The paper does not prescribe one concrete importance estimator, clustering
algorithm, or formula for choosing the number of representations. The gradient
L2 norm is therefore an explicit QA-domain instantiation of P, not an algorithm
claimed to be uniquely prescribed by the paper. This implementation uses
agglomerative clustering and selects the number of clusters by minimizing
cosine misalignment plus a configurable representation penalty, subject to a
minimum receiver-to-cluster alignment.
