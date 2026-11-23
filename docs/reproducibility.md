# Reproducibility record

## Released model

- Name: BSRT-A0
- Seed: `20261103`
- Frozen encoder: `deepset/bert-base-cased-squad2`
- Retention head: 320,353 parameters
- Hidden size: 96
- Transformer layers: 2
- Attention heads: 4
- Training contexts: 256
- Validation contexts: 12
- Best validation F1: 0.7423295
- Selected stage: answer-span pretraining, epoch 7

The original trusted training checkpoint had SHA-256:

```text
de6b4087bc6f69d828cfecf3b04d3038f3e33dbb22c4d33e4671c0381cbca043
```

The public tensor-only conversion `checkpoints/bsrt_a0.safetensors` has SHA-256:

```text
76b3f6220e50cc6e3e4befcfc364ee1da7d84d9b7dfb13c4d917310d219d30d5
```

Conversion preserved all 37 state-dictionary tensors. Optimizer state, absolute paths, and Python-pickled metadata were intentionally removed.

## Data boundaries

The project used SQuAD 2.0. Article-level splitting prevents contexts from one article appearing in more than one training or validation partition. Dataset files and embedding caches are excluded from this release.

The 500-example evaluation archive is a stable comparison cohort. It was repeatedly examined while the thesis was developed and is not described as untouched. A stronger follow-up evaluation would lock a new external cohort before any further modeling decisions.

## Metric artifacts

- `results/comparison_summary.csv` contains aggregate results across 5% to 50% removal
- `results/paired_f1_differences.csv` contains paired mean differences and context-cluster bootstrap intervals
- `results/ablation_metrics.csv` and `results/ABLATIONS.md` document controlled component ablations
- `results/training_summary.json` records training resource and selection statistics

The test suite verifies the published headline values directly from these immutable aggregate files.
