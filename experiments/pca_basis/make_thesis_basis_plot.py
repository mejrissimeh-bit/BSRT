from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = (
    HERE.parent
    / "results"
    / "pca_l2_context_retrieval"
    / "equivalent_5_to_50_sweep"
)


def main():
    frame = pd.read_csv(
        RESULTS / "equivalent_sweep_summary_with_coefficient_reduction.csv"
    )
    x = 100.0 * frame["actual_bit_reduction"]
    metrics = [
        ("accuracy", "Accuracy"),
        ("f1", "Answer F1"),
        ("semantic_similarity", "Semantic similarity"),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    # Each metric receives the same full text width for a fair, readable
    # comparison in the thesis.
    fig, axes = plt.subplots(3, 1, figsize=(6.7, 5.1), sharex=True)
    for axis, (column, label) in zip(axes, metrics):
        axis.plot(x, frame[column], color="#1d4ed8", marker="o", linewidth=2.4, markersize=5.5)
        axis.set_title(label, fontsize=11, fontweight="bold", pad=3)
        axis.set_ylim(0.75 if column != "semantic_similarity" else 0.80, 0.94)
        axis.set_yticks([0.75, 0.80, 0.85, 0.90, 0.94] if column != "semantic_similarity" else [0.80, 0.84, 0.88, 0.92, 0.94])
        axis.grid(True, color="#dbe3ee", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel("Score")
    axes[-1].set_xlabel("Bit reduction relative to original text (%)")
    fig.suptitle("PCA-Based Semantic Basis Compression", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.7)
    fig.savefig(RESULTS / "pca_basis_quality_thesis.pdf", bbox_inches="tight")
    fig.savefig(RESULTS / "pca_basis_quality_thesis.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
