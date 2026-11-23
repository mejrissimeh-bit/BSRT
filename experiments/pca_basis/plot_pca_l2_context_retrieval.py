from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results" / "pca_l2_context_retrieval"

METRICS = [
    ("retrieval_accuracy", "Retrieval Accuracy"),
    ("qa_accuracy", "QA Accuracy"),
    ("f1", "F1"),
    ("semantic_similarity", "Semantic Similarity"),
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_rows(results_dir: Path) -> list[dict]:
    path = results_dir / "semantic_basis_compression_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            row[key] = float(value)
    return rows


def text(x, y, value, size=13, anchor="middle", weight="400", color="#111827", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{transform}>{value}</text>'
    )


def sx(value, min_x, max_x, left, width):
    if min_x == max_x:
        return left + width / 2
    return left + (value - min_x) / (max_x - min_x) * width


def sy(value, min_y, max_y, top, height):
    if min_y == max_y:
        return top + height / 2
    return top + (max_y - value) / (max_y - min_y) * height


def write_svg(path: Path, width: int, height: int, body: list[str]):
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                *body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def plot_summary(rows: list[dict], output_path: Path):
    width, height = 1050, 660
    left, top, plot_w, plot_h = 90, 70, 760, 460
    x_values = [row["top_k_coefficients"] for row in rows]
    colors = {
        "retrieval_accuracy": "#111827",
        "qa_accuracy": "#2563eb",
        "f1": "#16a34a",
        "semantic_similarity": "#9333ea",
    }
    body = [
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 36, "Paper-Like Semantic Basis Compression", size=21, weight="700"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for x_value in x_values:
        x = sx(x_value, min(x_values), max(x_values), left, plot_w)
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e2e8f0"/>')
        body.append(text(x, top + plot_h + 28, f"{int(x_value)}", size=12))
    for tick in range(6):
        value = tick / 5
        y = sy(value, 0.0, 1.0, top, plot_h)
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        body.append(text(left - 10, y + 4, f"{value:.2f}", size=11, anchor="end"))

    for metric, label in METRICS:
        points = [
            (
                sx(row["top_k_coefficients"], min(x_values), max(x_values), left, plot_w),
                sy(row[metric], 0.0, 1.0, top, plot_h),
            )
            for row in sorted(rows, key=lambda item: item["top_k_coefficients"])
        ]
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        color = colors[metric]
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" stroke="white"/>')

    body.append(text(left + plot_w / 2, height - 48, "Transmitted top-k PCA coefficients", size=14))
    body.append(text(28, top + plot_h / 2, "Score", size=14, rotate=-90))
    legend_x = left + plot_w + 30
    body.append(text(legend_x, top + 8, "Metric", size=13, anchor="start", weight="700"))
    for index, (metric, label) in enumerate(METRICS):
        y = top + 34 + index * 28
        body.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{colors[metric]}" stroke-width="3"/>')
        body.append(text(legend_x + 38, y + 4, label, size=12, anchor="start"))
    write_svg(output_path, width, height, body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()
    results_dir = resolve_path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(results_dir)
    plot_summary(rows, plots_dir / "semantic_basis_compression_curves.svg")
    print(f"Plots written to: {plots_dir}")


if __name__ == "__main__":
    main()
