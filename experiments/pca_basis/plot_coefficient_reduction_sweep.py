from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
DEFAULT_SWEEP_DIR = (
    EXPERIMENT_DIR
    / "results"
    / "pca_l2_context_retrieval"
    / "equivalent_5_to_50_sweep"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP_DIR))
    parser.add_argument("--full-coefficients", type=int, default=256)
    return parser.parse_args()


def text(x, y, value, size=13, anchor="middle", weight="400", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#111827" '
        f'text-anchor="{anchor}"{transform}>{value}</text>'
    )


def write_plot(path: Path, rows: list[dict]) -> None:
    width, height = 980, 600
    left, top, plot_w, plot_h = 95, 70, 700, 410
    x_values = [row["coefficient_reduction_percent"] for row in rows]
    x_min, x_max = min(x_values), max(x_values)

    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="600" viewBox="0 0 980 600">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 36, "Semantic Basis Degradation by Coefficient Reduction", size=20, weight="700"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]

    for row in rows:
        x_value = row["coefficient_reduction_percent"]
        x = left + (x_value - x_min) / (x_max - x_min) * plot_w
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e2e8f0"/>')
        body.append(text(x, top + plot_h + 28, f"{x_value:.1f}%", size=10))

    for tick in range(6):
        y_value = tick / 5
        y = top + (1 - y_value) * plot_h
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        body.append(text(left - 12, y + 4, f"{y_value:.1f}", size=11, anchor="end"))

    curves = [
        ("retrieval_accuracy", "Retrieval accuracy", "#111827"),
        ("accuracy", "QA accuracy", "#2563eb"),
        ("f1", "F1", "#16a34a"),
        ("semantic_similarity", "Semantic similarity", "#9333ea"),
    ]
    for key, _label, color in curves:
        points = []
        for row in rows:
            x = left + (row["coefficient_reduction_percent"] - x_min) / (x_max - x_min) * plot_w
            y = top + (1 - row[key]) * plot_h
            points.append((x, y))
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="{color}" stroke="white"/>')

    body.append(text(left + plot_w / 2, height - 46, "Coefficient reduction from full 256-dimensional PCA code", size=14))
    body.append(text(28, top + plot_h / 2, "Score", size=14, rotate=-90))

    legend_x = left + plot_w + 28
    for index, (_key, label, color) in enumerate(curves):
        y = top + 28 + index * 28
        body.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(text(legend_x + 36, y + 4, label, size=12, anchor="start"))

    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    summary_path = sweep_dir / "equivalent_sweep_summary.csv"
    frame = pd.read_csv(summary_path)
    frame["coefficient_reduction_percent"] = (
        1.0 - frame["top_k_coefficients"] / float(args.full_coefficients)
    ) * 100.0
    frame.to_csv(sweep_dir / "equivalent_sweep_summary_with_coefficient_reduction.csv", index=False)
    write_plot(
        sweep_dir / "coefficient_reduction_degradation_curves.svg",
        frame.to_dict("records"),
    )
    print(f"Wrote coefficient-reduction plot to: {sweep_dir}")


if __name__ == "__main__":
    main()
