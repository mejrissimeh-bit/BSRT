from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
PROJECT_ROOT = EXPERIMENT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from main import evaluate_prediction  # noqa: E402


DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results" / "pca_l2_context_retrieval"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "equivalent_5_to_50_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--reductions", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--coefficient-bits", type=int, default=32)
    parser.add_argument("--index-bits", type=int, default=8)
    parser.add_argument("--n-components", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qa-model", default="deepset/bert-base-cased-squad2")
    parser.add_argument("--similarity-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_answers(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [str(value)]


def utf8_bits(text: str) -> int:
    return len(str(text).encode("utf-8")) * 8


def split_indices(count: int, train_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    train_count = max(2, int(round(count * train_fraction)))
    train_count = min(train_count, count - 1)
    return indices[:train_count], indices[train_count:]


def sparse_reconstruct(coefficients: np.ndarray, pca: PCA, top_k: int) -> np.ndarray:
    sparse = np.zeros_like(coefficients)
    if top_k >= coefficients.shape[1]:
        sparse[:] = coefficients
    else:
        top_indices = np.argpartition(np.abs(coefficients), -top_k, axis=1)[:, -top_k:]
        for row_index, indices in enumerate(top_indices):
            sparse[row_index, indices] = coefficients[row_index, indices]
    return sparse @ pca.components_ + pca.mean_


def choose_top_k(mean_original_bits: float, target_reduction: int, coeff_total_bits: int, max_components: int) -> int:
    target_bits = mean_original_bits * (1.0 - target_reduction / 100.0)
    return max(1, min(max_components, int(round(target_bits / coeff_total_bits))))


def append_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def text(x, y, value, size=13, anchor="middle", weight="400", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#111827" text-anchor="{anchor}"{transform}>{value}</text>'
    )


def write_plot(path: Path, rows: list[dict]) -> None:
    width, height = 980, 600
    left, top, plot_w, plot_h = 90, 70, 720, 410
    reductions = [row["target_bit_reduction_percent"] for row in rows]
    body = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="600" viewBox="0 0 980 600">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 36, "Semantic Basis Accuracy at Token-Reduction-Equivalent Bit Budgets", size=20, weight="700"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for r in reductions:
        x = left + (r - min(reductions)) / (max(reductions) - min(reductions)) * plot_w
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e2e8f0"/>')
        body.append(text(x, top + plot_h + 28, f"{int(r)}%", size=11))
    for tick in range(6):
        y_value = tick / 5
        y = top + (1 - y_value) * plot_h
        body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        body.append(text(left - 10, y + 4, f"{y_value:.1f}", size=11, anchor="end"))

    for key, label, color in [
        ("retrieval_accuracy", "Retrieval accuracy", "#111827"),
        ("accuracy", "QA accuracy", "#2563eb"),
        ("f1", "F1", "#16a34a"),
        ("semantic_similarity", "Semantic similarity", "#9333ea"),
    ]:
        points = []
        for row in rows:
            x = left + (row["target_bit_reduction_percent"] - min(reductions)) / (max(reductions) - min(reductions)) * plot_w
            y = top + (1 - row[key]) * plot_h
            points.append((x, y))
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="{color}" stroke="white"/>')
    body.append(text(left + plot_w / 2, height - 46, "Target bit reduction equivalent to token-removal levels", size=14))
    body.append(text(28, top + plot_h / 2, "Score", size=14, rotate=-90))
    legend_x = left + plot_w + 28
    for index, (_, label, color) in enumerate([
        ("retrieval_accuracy", "Retrieval accuracy", "#111827"),
        ("accuracy", "QA accuracy", "#2563eb"),
        ("f1", "F1", "#16a34a"),
        ("semantic_similarity", "Similarity", "#9333ea"),
    ]):
        y = top + 28 + index * 28
        body.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(text(legend_x + 36, y + 4, label, size=12, anchor="start"))
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = resolve(args.results_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors = np.load(results_dir / "l2_importance_vectors.npy")
    context_table = pd.read_csv(results_dir / "context_database.csv")
    train_indices, test_indices_all = split_indices(len(context_table), args.train_fraction, args.seed)
    test_indices = test_indices_all[: args.samples]
    n_components = min(args.n_components, len(train_indices), vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    pca.fit(vectors[train_indices])
    coefficients = pca.transform(vectors[test_indices])
    mean_original_bits = float(context_table.iloc[test_indices]["original_context"].map(utf8_bits).mean())
    coeff_total_bits = int(args.coefficient_bits) + int(args.index_bits)

    tokenizer = AutoTokenizer.from_pretrained(args.qa_model, local_files_only=args.local_files_only)
    model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model, local_files_only=args.local_files_only)
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
    similarity_model = SentenceTransformer(args.similarity_model, local_files_only=args.local_files_only)

    summary_rows = []
    sample_path = output_dir / "equivalent_sweep_sample_results.csv"
    if sample_path.exists():
        sample_path.unlink()
    for reduction in args.reductions:
        top_k = choose_top_k(mean_original_bits, int(reduction), coeff_total_bits, n_components)
        transmitted_bits = top_k * coeff_total_bits
        reconstructed = sparse_reconstruct(coefficients, pca, top_k)
        distances = euclidean_distances(reconstructed, vectors)
        retrieved = np.argmin(distances, axis=1)
        metric_rows = []
        print(f"Target {reduction}% bit reduction -> top_k={top_k}, bits={transmitted_bits}")
        for local_index, source_index in enumerate(test_indices):
            retrieved_index = int(retrieved[local_index])
            source = context_table.iloc[int(source_index)]
            candidate = context_table.iloc[retrieved_index]
            answers = parse_answers(source.true_answers)
            prediction = qa(question=str(source.question), context=str(candidate.original_context))["answer"]
            accuracy, exact_match, f1, similarity = evaluate_prediction(prediction, answers, similarity_model)
            row = {
                "target_bit_reduction_percent": int(reduction),
                "top_k_coefficients": int(top_k),
                "transmitted_bits": int(transmitted_bits),
                "source_context_id": int(source_index),
                "retrieved_context_id": int(retrieved_index),
                "retrieval_correct": int(source_index) == retrieved_index,
                "dataset_index": int(source.dataset_index),
                "prediction": prediction,
                "accuracy": accuracy,
                "exact_match": exact_match,
                "f1": f1,
                "semantic_similarity": similarity,
            }
            append_row(sample_path, row)
            metric_rows.append(row)
        frame = pd.DataFrame(metric_rows)
        summary_rows.append(
            {
                "target_bit_reduction_percent": int(reduction),
                "top_k_coefficients": int(top_k),
                "transmitted_bits": int(transmitted_bits),
                "mean_original_bits": mean_original_bits,
                "actual_bit_reduction": float(1.0 - transmitted_bits / mean_original_bits),
                "samples": len(frame),
                "retrieval_accuracy": float(frame["retrieval_correct"].mean()),
                "accuracy": float(frame["accuracy"].mean()),
                "exact_match": float(frame["exact_match"].mean()),
                "f1": float(frame["f1"].mean()),
                "semantic_similarity": float(frame["semantic_similarity"].mean()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "equivalent_sweep_summary.csv", index=False)
    write_plot(output_dir / "equivalent_5_to_50_accuracy_curves.svg", summary_rows)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Equivalent sweep written to: {output_dir}")


if __name__ == "__main__":
    main()
