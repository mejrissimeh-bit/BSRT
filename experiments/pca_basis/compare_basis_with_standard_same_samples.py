from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
DEFAULT_BASIS_RESULTS = EXPERIMENT_DIR / "results" / "pca_l2_context_retrieval"
DEFAULT_STANDARD_CSV = PROJECT_ROOT / "extended_training_sets" / "results" / "standard_methods_sample_results.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_BASIS_RESULTS / "same_100_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-results-dir", default=str(DEFAULT_BASIS_RESULTS))
    parser.add_argument("--standard-csv", default=str(DEFAULT_STANDARD_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--coefficient-bits", type=int, default=32)
    parser.add_argument("--index-bits", type=int, default=8)
    parser.add_argument("--removal-percentages", type=int, nargs="+", default=[30, 40, 50])
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def utf8_bits(text: str) -> int:
    return len(str(text).encode("utf-8")) * 8


def load_same_sample_ids(basis_dir: Path, top_k: int, sample_count: int) -> list[int]:
    qa_path = basis_dir / "qa_retrieval_sample_results.csv"
    if not qa_path.exists():
        raise FileNotFoundError(f"Missing basis QA results: {qa_path}")
    frame = pd.read_csv(qa_path, usecols=["top_k_coefficients", "dataset_index"])
    frame = frame[frame["top_k_coefficients"].astype(int) == int(top_k)]
    sample_ids = frame["dataset_index"].dropna().astype(int).drop_duplicates().tolist()
    if len(sample_ids) < sample_count:
        raise RuntimeError(f"Only found {len(sample_ids)} basis samples for top-k={top_k}.")
    return sample_ids[:sample_count]


def summarize_original(standard: pd.DataFrame, sample_ids: list[int]) -> dict:
    base = (
        standard[standard["dataset_index"].astype(int).isin(set(sample_ids))]
        .drop_duplicates("dataset_index")
        .copy()
    )
    original_bits = base["original_context"].map(utf8_bits)
    return {
        "strategy": "Original context",
        "method": "original",
        "token_removal_percent": 0,
        "samples": len(base),
        "mean_transmitted_bits": float(original_bits.mean()),
        "mean_bit_reduction_vs_original": 0.0,
        "accuracy": float(base["original_accuracy"].mean()),
        "exact_match": float(base["original_exact_match"].mean()),
        "f1": float(base["original_f1"].mean()),
        "semantic_similarity": float(base["original_semantic_similarity"].mean()),
    }


def summarize_standard(standard: pd.DataFrame, sample_ids: list[int], percentages: list[int]) -> list[dict]:
    subset = standard[
        standard["dataset_index"].astype(int).isin(set(sample_ids))
        & standard["removal_percent"].astype(int).isin(set(percentages))
    ].copy()
    rows = []
    for (method, percent), group in subset.groupby(["method", "removal_percent"]):
        compressed_bits = group["compressed_context"].map(utf8_bits)
        original_bits = group["original_context"].map(utf8_bits)
        rows.append(
            {
                "strategy": f"{method.upper()} {int(percent)}%",
                "method": method,
                "token_removal_percent": int(percent),
                "samples": len(group),
                "mean_transmitted_bits": float(compressed_bits.mean()),
                "mean_bit_reduction_vs_original": float(1.0 - (compressed_bits / original_bits).mean()),
                "accuracy": float(group["accuracy"].mean()),
                "exact_match": float(group["exact_match"].mean()),
                "f1": float(group["f1"].mean()),
                "semantic_similarity": float(group["semantic_similarity"].mean()),
            }
        )
    return sorted(rows, key=lambda row: (row["token_removal_percent"], row["method"]))


def summarize_basis(basis_dir: Path, sample_ids: list[int], top_k: int, coefficient_bits: int, index_bits: int, standard: pd.DataFrame) -> dict:
    qa_path = basis_dir / "qa_retrieval_sample_results.csv"
    basis = pd.read_csv(qa_path)
    basis = basis[
        basis["top_k_coefficients"].astype(int).eq(int(top_k))
        & basis["dataset_index"].astype(int).isin(set(sample_ids))
    ].copy()
    basis = basis.drop_duplicates("dataset_index")
    base_contexts = (
        standard[standard["dataset_index"].astype(int).isin(set(sample_ids))]
        .drop_duplicates("dataset_index")
        .set_index("dataset_index")
    )
    original_bits = base_contexts.loc[basis["dataset_index"].astype(int), "original_context"].map(utf8_bits).reset_index(drop=True)
    transmitted_bits = int(top_k) * (int(coefficient_bits) + int(index_bits))
    return {
        "strategy": f"Semantic basis top-{top_k}",
        "method": "semantic_basis",
        "token_removal_percent": None,
        "samples": len(basis),
        "mean_transmitted_bits": float(transmitted_bits),
        "mean_bit_reduction_vs_original": float(1.0 - transmitted_bits / original_bits.mean()),
        "accuracy": float(basis["accuracy"].mean()),
        "exact_match": float(basis["exact_match"].mean()),
        "f1": float(basis["f1"].mean()),
        "semantic_similarity": float(basis["semantic_similarity"].mean()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], args: argparse.Namespace, sample_ids: list[int]) -> None:
    lines = [
        "# Same-Sample Comparison: Semantic Basis vs Standard Compression",
        "",
        f"- Same samples: {len(sample_ids)}",
        f"- Semantic basis setting: top-{args.top_k} PCA coefficients",
        f"- Coefficient bit estimate: {args.coefficient_bits} bits/value + {args.index_bits} bits/index",
        f"- Standard baselines: Random, L2, Attention at {args.removal_percentages}% token removal",
        "",
        "The semantic-basis method transmits sparse PCA coefficients of an L2-importance vector, not text.",
        "The receiver retrieves the closest context from a shared candidate database.",
        "",
        "| Strategy | Token removal | Mean bits | Bit reduction vs original | Accuracy | EM | F1 | Similarity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        token_removal = (
            "N/A"
            if pd.isna(row["token_removal_percent"])
            else f"{int(row['token_removal_percent'])}%"
        )
        lines.append(
            f"| {row['strategy']} | {token_removal} | {row['mean_transmitted_bits']:.1f} | "
            f"{row['mean_bit_reduction_vs_original']:.3f} | {row['accuracy']:.3f} | "
            f"{row['exact_match']:.3f} | {row['f1']:.3f} | {row['semantic_similarity']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Token removal is only defined for text-token removal baselines.",
            "- Semantic basis compression is measured by transmitted coefficient bits, not token-removal percentage.",
            "- Lower mean bits / higher bit reduction means stronger communication compression.",
            "- Higher Accuracy/F1/Similarity means better QA preservation.",
            "- The semantic-basis method is only directly comparable under the shared-database assumption.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_text(x, y, value, size=12, anchor="middle", weight="400", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#111827" text-anchor="{anchor}"{transform}>{value}</text>'
    )


def write_svg_plot(path: Path, rows: list[dict]) -> None:
    selected = [
        row for row in rows
        if row["method"] in {"original", "l2", "attention", "semantic_basis"}
        and (
            pd.isna(row["token_removal_percent"])
            or int(row["token_removal_percent"]) in {0, 40, 50}
        )
    ]
    width, height = 1160, 620
    left, top, plot_w, plot_h = 90, 70, 760, 410
    max_bits = max(row["mean_transmitted_bits"] for row in selected) * 1.08
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="620" viewBox="0 0 1160 620">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 36, "Same 100 Samples: Bit Cost vs QA Accuracy", size=21, weight="700"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for tick in range(6):
        x_value = max_bits * tick / 5
        x = left + x_value / max_bits * plot_w
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e2e8f0"/>')
        body.append(svg_text(x, top + plot_h + 28, f"{x_value:.0f}", size=11))
        y_value = tick / 5
        y = top + (1 - y_value) * plot_h
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        body.append(svg_text(left - 10, y + 4, f"{y_value:.1f}", size=11, anchor="end"))
    colors = {
        "original": "#111827",
        "l2": "#2563eb",
        "attention": "#f59e0b",
        "semantic_basis": "#16a34a",
    }
    for row in selected:
        x = left + row["mean_transmitted_bits"] / max_bits * plot_w
        y = top + (1 - row["accuracy"]) * plot_h
        color = colors.get(row["method"], "#64748b")
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="white" stroke-width="1.5"/>')
        body.append(svg_text(x + 10, y - 10, row["strategy"], size=10, anchor="start"))
    body.append(svg_text(left + plot_w / 2, height - 46, "Mean transmitted bits", size=14))
    body.append(svg_text(28, top + plot_h / 2, "Accuracy", size=14, rotate=-90))
    body.append(svg_text(left + plot_w / 2, height - 18, "Lower-left is cheaper; upper-left is better compression with high QA accuracy.", size=12))
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = parse_args()
    basis_dir = resolve(args.basis_results_dir)
    standard_csv = resolve(args.standard_csv)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = load_same_sample_ids(basis_dir, args.top_k, args.samples)
    standard = pd.read_csv(
        standard_csv,
        usecols=[
            "dataset_index",
            "method",
            "removal_percent",
            "original_context",
            "compressed_context",
            "original_accuracy",
            "original_exact_match",
            "original_f1",
            "original_semantic_similarity",
            "accuracy",
            "exact_match",
            "f1",
            "semantic_similarity",
        ],
    )
    rows = [summarize_original(standard, sample_ids)]
    rows.extend(summarize_standard(standard, sample_ids, args.removal_percentages))
    rows.append(
        summarize_basis(
            basis_dir,
            sample_ids,
            args.top_k,
            args.coefficient_bits,
            args.index_bits,
            standard,
        )
    )
    rows = sorted(
        rows,
        key=lambda row: (
            row["method"] == "semantic_basis",
            999 if pd.isna(row["token_removal_percent"]) else int(row["token_removal_percent"]),
            row["method"],
        ),
    )
    write_csv(output_dir / "same_100_comparison_summary.csv", rows)
    write_report(output_dir / "same_100_comparison_report.md", rows, args, sample_ids)
    write_svg_plot(output_dir / "same_100_bits_vs_accuracy.svg", rows)
    pd.DataFrame({"dataset_index": sample_ids}).to_csv(output_dir / "same_100_sample_ids.csv", index=False)
    print(f"Same-sample comparison written to: {output_dir}")


if __name__ == "__main__":
    main()
