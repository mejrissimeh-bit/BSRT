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
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoModel, AutoModelForQuestionAnswering, AutoTokenizer, pipeline


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
PROJECT_ROOT = EXPERIMENT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from main import evaluate_prediction  # noqa: E402


DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "pca_l2_context_retrieval.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-like semantic basis compression using L2 importance vectors and PCA retrieval."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--vector-length", type=int)
    parser.add_argument("--components", type=int)
    parser.add_argument("--top-k", type=int, nargs="+")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-vectors", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.max_contexts is not None:
        config["max_contexts"] = args.max_contexts
    if args.vector_length is not None:
        config["vector_length"] = args.vector_length
    if args.components is not None:
        config["n_components"] = args.components
    if args.top_k is not None:
        config["top_k_coefficients"] = args.top_k
    if args.skip_qa:
        config["evaluate_qa"] = False
    return config


def resolve_project_path(path_text: str) -> Path:
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


def load_context_table(source_csv: Path, max_contexts: int) -> pd.DataFrame:
    frame = pd.read_csv(
        source_csv,
        usecols=["dataset_index", "squad_id", "question", "true_answers", "original_context"],
    )
    frame = frame.dropna(subset=["original_context", "question", "true_answers"])
    frame = frame.drop_duplicates("original_context").sort_values("dataset_index")
    frame = frame.head(int(max_contexts)).reset_index(drop=True)
    frame["context_id"] = np.arange(len(frame), dtype=int)
    return frame


def compute_l2_importance_vector(context: str, tokenizer, model, vector_length: int) -> np.ndarray:
    encoded = tokenizer(
        context,
        return_tensors="pt",
        truncation=True,
        max_length=int(vector_length) + 2,
        padding=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state[0]
    input_ids = encoded["input_ids"][0]
    special_ids = {
        token_id
        for token_id in [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ]
        if token_id is not None
    }
    keep_positions = [
        index for index, token_id in enumerate(input_ids.tolist())
        if token_id not in special_ids
    ]
    token_vectors = hidden[keep_positions]
    norms = torch.linalg.vector_norm(token_vectors, ord=2, dim=1).detach().cpu().numpy()
    vector = np.zeros(int(vector_length), dtype=np.float32)
    limit = min(len(norms), int(vector_length))
    if limit:
        vector[:limit] = norms[:limit]
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def build_or_load_vectors(config: dict, context_table: pd.DataFrame, output_dir: Path, rebuild: bool) -> np.ndarray:
    vector_path = output_dir / "l2_importance_vectors.npy"
    meta_path = output_dir / "l2_importance_vectors_metadata.json"
    expected_meta = {
        "source_csv": str(resolve_project_path(config["source_csv"])),
        "max_contexts": int(config["max_contexts"]),
        "vector_length": int(config["vector_length"]),
        "bert_model": config["bert_model"],
    }
    if vector_path.exists() and meta_path.exists() and not rebuild:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata == expected_meta:
            print(f"Loading cached L2 vectors: {vector_path}")
            return np.load(vector_path)

    print("Computing BERT L2 importance vectors...")
    local_files_only = bool(config.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(config["bert_model"], local_files_only=local_files_only)
    model = AutoModel.from_pretrained(config["bert_model"], local_files_only=local_files_only)
    model.eval()
    vectors = []
    for position, row in enumerate(context_table.itertuples(index=False), start=1):
        vectors.append(
            compute_l2_importance_vector(
                str(row.original_context),
                tokenizer,
                model,
                int(config["vector_length"]),
            )
        )
        if position % 25 == 0 or position == len(context_table):
            print(f"  vectors {position}/{len(context_table)}")
    matrix = np.vstack(vectors).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, matrix)
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return matrix


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


def retrieve_contexts(reconstructed: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = euclidean_distances(reconstructed, candidates)
    retrieved = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(retrieved)), retrieved]
    return retrieved, best_distances


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_qa_for_retrievals(
    config: dict,
    context_table: pd.DataFrame,
    test_indices: np.ndarray,
    retrieved_indices_by_k: dict[int, np.ndarray],
    output_dir: Path,
) -> pd.DataFrame:
    local_files_only = bool(config.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(config["qa_model"], local_files_only=local_files_only)
    model = AutoModelForQuestionAnswering.from_pretrained(config["qa_model"], local_files_only=local_files_only)
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
    similarity_model = SentenceTransformer(config["similarity_model"], local_files_only=local_files_only)

    rows = []
    output_path = output_dir / "qa_retrieval_sample_results.csv"
    completed = set()
    if output_path.exists():
        previous = pd.read_csv(output_path)
        rows = previous.to_dict("records")
        completed = {
            (int(row.top_k_coefficients), int(row.source_context_id))
            for row in previous.itertuples(index=False)
        }

    for top_k, retrieved_indices in retrieved_indices_by_k.items():
        print(f"QA evaluation for top-k={top_k}")
        for local_index, source_index in enumerate(test_indices):
            key = (int(top_k), int(source_index))
            if key in completed:
                continue
            retrieved_index = int(retrieved_indices[local_index])
            source_row = context_table.iloc[int(source_index)]
            retrieved_row = context_table.iloc[retrieved_index]
            true_answers = parse_answers(source_row.true_answers)
            prediction = qa(
                question=str(source_row.question),
                context=str(retrieved_row.original_context),
            )["answer"]
            accuracy, exact_match, f1, similarity = evaluate_prediction(
                prediction,
                true_answers,
                similarity_model,
            )
            row = {
                "top_k_coefficients": int(top_k),
                "source_context_id": int(source_index),
                "retrieved_context_id": int(retrieved_index),
                "retrieval_correct": int(source_index) == int(retrieved_index),
                "dataset_index": int(source_row.dataset_index),
                "squad_id": source_row.squad_id,
                "question": source_row.question,
                "true_answers": json.dumps(true_answers, ensure_ascii=False),
                "prediction": prediction,
                "accuracy": accuracy,
                "exact_match": exact_match,
                "f1": f1,
                "semantic_similarity": similarity,
            }
            append_row(output_path, row)
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(output_dir: Path, config: dict, summary: pd.DataFrame) -> None:
    lines = [
        "# Paper-Like Semantic Basis Compression Prototype",
        "",
        "This experiment adapts the paper's basis-compression idea to text contexts.",
        "The transmitted object is not the context text, but sparse PCA coefficients of an L2 token-importance vector.",
        "",
        f"- Source CSV: `{config['source_csv']}`",
        f"- Max contexts: {config['max_contexts']}",
        f"- Vector length: {config['vector_length']}",
        f"- PCA components: {config['n_components']}",
        f"- Top-k coefficients: {config['top_k_coefficients']}",
        "",
        "| Top-k coeffs | Retrieval acc. | Mean distance | QA acc. | EM | F1 | Similarity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.top_k_coefficients)} | {row.retrieval_accuracy:.4f} | "
            f"{row.mean_retrieval_distance:.4f} | {row.qa_accuracy:.4f} | "
            f"{row.exact_match:.4f} | {row.f1:.4f} | {row.semantic_similarity:.4f} |"
        )
    (output_dir / "semantic_basis_compression_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args)
    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source_csv = resolve_project_path(config["source_csv"])
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {source_csv}")

    context_table = load_context_table(source_csv, int(config["max_contexts"]))
    top_k_values = [int(value) for value in config["top_k_coefficients"]]
    plan = {
        "contexts": len(context_table),
        "vector_length": int(config["vector_length"]),
        "n_components": int(config["n_components"]),
        "top_k_coefficients": top_k_values,
        "train_fraction": float(config["train_fraction"]),
        "evaluate_qa": bool(config.get("evaluate_qa", True)),
    }
    (output_dir / "run_config.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("Dry run complete.")
        return

    vectors = build_or_load_vectors(config, context_table, output_dir, args.rebuild_vectors)
    train_indices, test_indices = split_indices(
        len(context_table),
        float(config["train_fraction"]),
        int(config["random_seed"]),
    )
    n_components = min(int(config["n_components"]), len(train_indices), vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=int(config["random_seed"]))
    pca.fit(vectors[train_indices])
    coefficients = pca.transform(vectors[test_indices])

    rows = []
    retrieved_by_k = {}
    for top_k in top_k_values:
        effective_top_k = min(int(top_k), n_components)
        reconstructed = sparse_reconstruct(coefficients, pca, effective_top_k)
        retrieved_indices, distances = retrieve_contexts(reconstructed, vectors)
        retrieved_by_k[int(top_k)] = retrieved_indices
        correct = retrieved_indices == test_indices
        rows.append(
            {
                "top_k_coefficients": int(top_k),
                "effective_top_k": int(effective_top_k),
                "test_contexts": int(len(test_indices)),
                "retrieval_accuracy": float(np.mean(correct)),
                "mean_retrieval_distance": float(np.mean(distances)),
                "qa_accuracy": 0.0,
                "exact_match": 0.0,
                "f1": 0.0,
                "semantic_similarity": 0.0,
            }
        )
    summary = pd.DataFrame(rows)

    if bool(config.get("evaluate_qa", True)):
        qa_rows = evaluate_qa_for_retrievals(
            config,
            context_table,
            test_indices,
            retrieved_by_k,
            output_dir,
        )
        qa_summary = (
            qa_rows.groupby("top_k_coefficients")
            .agg(
                qa_accuracy=("accuracy", "mean"),
                exact_match=("exact_match", "mean"),
                f1=("f1", "mean"),
                semantic_similarity=("semantic_similarity", "mean"),
            )
            .reset_index()
        )
        summary = summary.drop(columns=["qa_accuracy", "exact_match", "f1", "semantic_similarity"])
        summary = summary.merge(qa_summary, on="top_k_coefficients", how="left")

    summary.to_csv(output_dir / "semantic_basis_compression_summary.csv", index=False)
    context_table.to_csv(output_dir / "context_database.csv", index=False)
    write_report(output_dir, config, summary)
    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
