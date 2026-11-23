from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PCA_SWEEP_DIR = (
    ROOT
    / "semantic_basis_compressor"
    / "results"
    / "pca_l2_context_retrieval"
    / "equivalent_5_to_50_sweep"
)
QPSK_DIR = ROOT / "semantic_basis_advanced_channel" / "results" / "same_100_qam_awgn"


def text(x, y, value, size=24, anchor="middle", weight="400", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#111827" '
        f'text-anchor="{anchor}"{transform}>{value}</text>'
    )


def sx(value, min_x, max_x, left, width):
    return left + (float(value) - min_x) / (max_x - min_x) * width


def sy(value, min_y, max_y, top, height):
    return top + (max_y - float(value)) / (max_y - min_y) * height


def polyline(points, color, width=5, dash=None):
    dash_text = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{dash_text}/>'


def write_clean_no_channel_plot() -> None:
    frame = pd.read_csv(PCA_SWEEP_DIR / "equivalent_sweep_summary_with_coefficient_reduction.csv")
    frame["actual_bit_reduction_percent"] = frame["actual_bit_reduction"].astype(float) * 100.0

    width, height = 1500, 860
    left, top, plot_w, plot_h = 155, 105, 1080, 560
    x_min, x_max = 0, 52
    y_min, y_max = 0.78, 1.01

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 55, "PCA Semantic-Basis Test Without Channel", 32, weight="700"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>',
    ]

    for tick in [0, 10, 20, 30, 40, 50]:
        x = sx(tick, x_min, x_max, left, plot_w)
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e2e8f0" stroke-width="2"/>')
        body.append(text(x, top + plot_h + 50, f"{tick}%", 22))

    for tick in [0.80, 0.85, 0.90, 0.95, 1.00]:
        y = sy(tick, y_min, y_max, top, plot_h)
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="2"/>')
        body.append(text(left - 25, y + 8, f"{tick:.2f}", 22, anchor="end"))

    curves = [
        ("retrieval_accuracy", "Context retrieval", "#111827", None),
        ("accuracy", "QA accuracy", "#2563eb", None),
        ("f1", "F1", "#16a34a", "12 9"),
        ("semantic_similarity", "Semantic similarity", "#9333ea", "5 8"),
    ]
    for key, label, color, dash in curves:
        points = [
            (
                sx(row.actual_bit_reduction_percent, x_min, x_max, left, plot_w),
                sy(row._asdict()[key], y_min, y_max, top, plot_h),
            )
            for row in frame.itertuples(index=False)
        ]
        body.append(polyline(points, color, dash=dash))
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')

    legend_x = left + plot_w + 48
    for i, (_key, label, color, dash) in enumerate(curves):
        y = top + 55 + i * 58
        body.append(polyline([(legend_x, y), (legend_x + 50, y)], color, dash=dash))
        body.append(text(legend_x + 68, y + 8, label, 22, anchor="start"))

    body.append(text(left + plot_w / 2, height - 65, "Actual transmitted-bit reduction", 26, weight="700"))
    body.append(text(52, top + plot_h / 2, "Score", 26, weight="700", rotate=-90))
    body.append("</svg>")
    (PCA_SWEEP_DIR / "pca_no_channel_presentation.svg").write_text("\n".join(body), encoding="utf-8")


def write_clean_qpsk_plot() -> None:
    frame = pd.read_csv(QPSK_DIR / "semantic_basis_qam_awgn_summary.csv")
    frame = frame[frame["modulation"].astype(str).str.lower() == "qpsk"].copy()
    frame = frame[frame["target_bit_reduction_percent"].astype(int).isin([5, 20, 40, 50])]

    width, height = 1500, 860
    left, top, plot_w, plot_h = 155, 105, 1080, 560
    x_min, x_max = 0, 15
    y_min, y_max = 0.0, 1.0

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 55, "Semantic-Basis Accuracy over QPSK AWGN Channel", 32, weight="700"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>',
    ]

    for tick in [0, 3, 5, 7, 10, 15]:
        x = sx(tick, x_min, x_max, left, plot_w)
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e2e8f0" stroke-width="2"/>')
        body.append(text(x, top + plot_h + 50, f"{tick:g}", 22))

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = sy(tick, y_min, y_max, top, plot_h)
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="2"/>')
        body.append(text(left - 25, y + 8, f"{tick:.1f}", 22, anchor="end"))

    colors = {5: "#16a34a", 20: "#dc2626", 40: "#7c3aed", 50: "#111827"}
    for reduction in [5, 20, 40, 50]:
        curve = frame[frame["target_bit_reduction_percent"].astype(int) == reduction].sort_values("snr_db")
        points = [
            (sx(row.snr_db, x_min, x_max, left, plot_w), sy(row.accuracy, y_min, y_max, top, plot_h))
            for row in curve.itertuples(index=False)
        ]
        color = colors[reduction]
        body.append(polyline(points, color))
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')

    legend_x = left + plot_w + 48
    for i, reduction in enumerate([5, 20, 40, 50]):
        y = top + 55 + i * 58
        color = colors[reduction]
        body.append(polyline([(legend_x, y), (legend_x + 50, y)], color))
        body.append(text(legend_x + 68, y + 8, f"{reduction}% reduction", 22, anchor="start"))

    body.append(text(left + plot_w / 2, height - 65, "SNR (dB)", 26, weight="700"))
    body.append(text(52, top + plot_h / 2, "QA accuracy", 26, weight="700", rotate=-90))
    body.append("</svg>")
    (QPSK_DIR / "plots" / "semantic_basis_qpsk_presentation.svg").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    write_clean_no_channel_plot()
    write_clean_qpsk_plot()
    print("Wrote presentation plots:")
    print(PCA_SWEEP_DIR / "pca_no_channel_presentation.svg")
    print(QPSK_DIR / "plots" / "semantic_basis_qpsk_presentation.svg")


if __name__ == "__main__":
    main()
