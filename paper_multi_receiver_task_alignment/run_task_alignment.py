from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_CONFIG = HERE / "config.json"

sys.path.insert(0, str(PROJECT_ROOT))
from main import evaluate_prediction  # noqa: E402


STRATEGIES = {
    "global_alignment": "Global semantic alignment",
    "task_alignment": "Clustered task alignment",
    "individual_tasks": "Individual task reference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-style semantic task alignment for multiple QA receivers."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def stable_context_id(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()[:16]


def parse_answers(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [str(value)]


def load_receivers(config: dict, max_contexts: int | None) -> pd.DataFrame:
    path = resolve(config["input_csv"])
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    required = {
        "dataset_index",
        "squad_id",
        "question",
        "true_answers",
        "original_context",
    }
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    frame = frame.drop_duplicates("dataset_index").copy()
    frame["context_id"] = frame["original_context"].map(stable_context_id)
    counts = frame.groupby("context_id")["question"].nunique()
    valid = set(counts[counts >= int(config["minimum_receivers_per_context"])].index)
    frame = frame[frame["context_id"].isin(valid)].copy()
    ordered_contexts = sorted(frame["context_id"].unique())
    if max_contexts is not None:
        ordered_contexts = ordered_contexts[: max(0, int(max_contexts))]
        frame = frame[frame["context_id"].isin(set(ordered_contexts))].copy()
    return frame.sort_values(["context_id", "dataset_index"]).reset_index(drop=True)


def make_qa_pipeline(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=True)
    device_id = 0 if torch.cuda.is_available() else -1
    if device_id >= 0:
        model = model.cuda()
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device_id)
    return tokenizer, model, qa


def qa_predictions(qa, questions: list[str], contexts: list[str], batch_size: int) -> list[dict]:
    inputs = [{"question": question, "context": context} for question, context in zip(questions, contexts)]
    outputs = qa(inputs, batch_size=int(batch_size))
    if isinstance(outputs, dict):
        outputs = [outputs]
    return list(outputs)


def token_cost_bits(tokenizer, token_id: int) -> int:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    return max(8, len(text.encode("utf-8", errors="replace")) * 8)


def relevance_cache_path(output_dir: Path, context_id: str, config: dict) -> Path:
    configured = config.get("relevance_cache_dir")
    cache_dir = resolve(configured) if configured else output_dir / "relevance_cache"
    return cache_dir / f"{context_id}.npz"


def find_answer_span(context: str, answers: list[str]) -> tuple[int, int] | None:
    lowered = context.lower()
    for answer in answers:
        answer = str(answer).strip()
        if not answer:
            continue
        start = context.find(answer)
        if start < 0:
            start = lowered.find(answer.lower())
        if start >= 0:
            return start, start + len(answer)
    return None


def align_context_scores(
    canonical_ids: list[int],
    paired_context_ids: list[int],
    paired_scores: np.ndarray,
    fallback_scores: np.ndarray,
) -> np.ndarray:
    if canonical_ids == paired_context_ids:
        return paired_scores.astype(float)
    aligned = fallback_scores.astype(float).copy()
    matcher = SequenceMatcher(a=canonical_ids, b=paired_context_ids, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            aligned[block.a + offset] = float(paired_scores[block.b + offset])
    return aligned


def task_distortion_gradient_l2(
    question: str,
    answers: list[str],
    context: str,
    canonical_ids: list[int],
    tokenizer,
    qa_model,
) -> np.ndarray:
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=min(int(tokenizer.model_max_length), 512),
    )
    sequence_ids = encoded.sequence_ids(0)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    device = next(qa_model.parameters()).device
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    input_ids = model_inputs.pop("input_ids")
    embeddings = qa_model.get_input_embeddings()(input_ids).detach()
    embeddings.requires_grad_(True)

    answer_span = find_answer_span(context, answers)
    start_position = end_position = None
    if answer_span is not None:
        answer_start, answer_end = answer_span
        candidates = [
            index
            for index, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1
            and offsets[index][1] > answer_start
            and offsets[index][0] < answer_end
        ]
        if candidates:
            start_position, end_position = candidates[0], candidates[-1]

    qa_model.eval()
    with torch.enable_grad():
        if start_position is not None:
            outputs = qa_model(
                inputs_embeds=embeddings,
                start_positions=torch.tensor([start_position], device=device),
                end_positions=torch.tensor([end_position], device=device),
                **model_inputs,
            )
            objective = outputs.loss
        else:
            outputs = qa_model(inputs_embeds=embeddings, **model_inputs)
            predicted_start = int(outputs.start_logits.argmax(dim=-1).item())
            predicted_end = int(outputs.end_logits.argmax(dim=-1).item())
            objective = -(
                outputs.start_logits[0, predicted_start]
                + outputs.end_logits[0, predicted_end]
            )
        gradients = torch.autograd.grad(objective, embeddings, retain_graph=False)[0][0]

    gradient_norms = torch.norm(gradients, p=2, dim=1).detach().cpu().numpy()
    all_ids = input_ids[0].detach().cpu().tolist()
    context_positions = [index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
    paired_context_ids = [int(all_ids[index]) for index in context_positions]
    paired_scores = gradient_norms[context_positions]
    with torch.no_grad():
        canonical_tensor = torch.tensor(canonical_ids, dtype=torch.long, device=device)
        fallback = torch.norm(qa_model.get_input_embeddings()(canonical_tensor), p=2, dim=1)
    return align_context_scores(
        canonical_ids,
        paired_context_ids,
        paired_scores,
        fallback.detach().cpu().numpy(),
    )


def compute_relevance_for_context(
    group: pd.DataFrame,
    tokenizer,
    qa_model,
    config: dict,
    output_dir: Path,
) -> dict:
    if config.get("importance_estimator") != "distortion_gradient_l2":
        raise ValueError(
            "This implementation currently supports "
            "importance_estimator='distortion_gradient_l2' only"
        )
    context_id = str(group.iloc[0]["context_id"])
    cache_path = relevance_cache_path(output_dir, context_id, config)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return {key: cached[key] for key in cached.files}

    original_context = str(group.iloc[0]["original_context"])
    token_ids = tokenizer(
        original_context,
        add_special_tokens=False,
        truncation=True,
        max_length=int(config["max_context_tokens"]),
    )["input_ids"]
    if len(token_ids) < 2:
        raise ValueError(f"Context {context_id} has fewer than two usable tokens")
    canonical_context = tokenizer.decode(token_ids, skip_special_tokens=True)
    receiver_ids = group["dataset_index"].astype(int).to_numpy()
    questions = group["question"].astype(str).tolist()
    answers = [parse_answers(value) for value in group["true_answers"]]
    costs = np.asarray([token_cost_bits(tokenizer, token_id) for token_id in token_ids], dtype=float)

    relevance = np.zeros((len(questions), len(token_ids)), dtype=float)
    for receiver_index, (question, refs) in enumerate(zip(questions, answers)):
        relevance[receiver_index] = task_distortion_gradient_l2(
            question,
            refs,
            canonical_context,
            token_ids,
            tokenizer,
            qa_model,
        )

    density = relevance / costs[np.newaxis, :]
    for row_index in range(len(density)):
        if not np.any(density[row_index] > 0):
            density[row_index] = 1.0 / costs

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        context_id=np.asarray(context_id),
        original_context=np.asarray(original_context),
        canonical_context=np.asarray(canonical_context),
        receiver_ids=receiver_ids,
        questions=np.asarray(questions, dtype=object),
        true_answers=np.asarray([json.dumps(item) for item in answers], dtype=object),
        token_ids=np.asarray(token_ids, dtype=int),
        token_text=np.asarray(tokenizer.convert_ids_to_tokens(token_ids), dtype=object),
        token_cost_bits=costs,
        relevance=relevance,
        density=density,
        importance_estimator=np.asarray(config["importance_estimator"]),
    )
    return {
        "context_id": np.asarray(context_id),
        "original_context": np.asarray(original_context),
        "canonical_context": np.asarray(canonical_context),
        "receiver_ids": receiver_ids,
        "questions": np.asarray(questions, dtype=object),
        "true_answers": np.asarray([json.dumps(item) for item in answers], dtype=object),
        "token_ids": np.asarray(token_ids, dtype=int),
        "token_text": np.asarray(tokenizer.convert_ids_to_tokens(token_ids), dtype=object),
        "token_cost_bits": costs,
        "relevance": relevance,
        "density": density,
        "importance_estimator": np.asarray(config["importance_estimator"]),
    }


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def co_priority(normalized_vectors: np.ndarray) -> np.ndarray:
    direction = normalized_vectors.sum(axis=0)
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return np.ones(normalized_vectors.shape[1], dtype=float) / math.sqrt(normalized_vectors.shape[1])
    return direction / norm


def angular_misalignment(normalized_vectors: np.ndarray, priority: np.ndarray) -> float:
    similarities = np.clip(normalized_vectors @ priority, -1.0, 1.0)
    return float(np.mean(np.arccos(similarities)))


def angular_co_priority(
    normalized_vectors: np.ndarray,
    maximum_iterations: int = 300,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Minimize mean angular distance on the nonnegative unit sphere."""
    priority = co_priority(normalized_vectors)
    objective = angular_misalignment(normalized_vectors, priority)

    for _ in range(maximum_iterations):
        similarities = np.clip(normalized_vectors @ priority, -1.0 + 1e-9, 1.0 - 1e-9)
        weights = 1.0 / np.sqrt(1.0 - similarities**2)
        ambient_gradient = -(normalized_vectors * weights[:, np.newaxis]).mean(axis=0)
        tangent_gradient = ambient_gradient - priority * float(priority @ ambient_gradient)
        gradient_norm = float(np.linalg.norm(tangent_gradient))
        if gradient_norm <= tolerance:
            break

        step = min(0.25, 0.1 / gradient_norm)
        accepted = False
        for _ in range(25):
            candidate = np.maximum(priority - step * tangent_gradient, 0.0)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > 1e-12:
                candidate /= candidate_norm
                candidate_objective = angular_misalignment(normalized_vectors, candidate)
                if candidate_objective <= objective - 1e-12:
                    accepted = True
                    break
            step *= 0.5

        if not accepted:
            break
        if abs(objective - candidate_objective) <= tolerance:
            priority = candidate
            break
        priority = candidate
        objective = candidate_objective

    return priority


def clustering_loss(
    normalized_vectors: np.ndarray,
    labels: np.ndarray,
    priority_objective: str = "cosine",
) -> tuple[float, dict[int, np.ndarray]]:
    priorities = {}
    losses = []
    for cluster in sorted(set(labels.tolist())):
        members = normalized_vectors[labels == cluster]
        if priority_objective == "cosine":
            priority = co_priority(members)
        elif priority_objective == "angular":
            priority = angular_co_priority(members)
        else:
            raise ValueError(f"Unknown co-priority objective: {priority_objective}")
        priorities[int(cluster)] = priority
        similarities = np.clip(members @ priority, -1.0, 1.0)
        if priority_objective == "angular":
            losses.extend(np.arccos(similarities).tolist())
        else:
            losses.extend((1.0 - similarities).tolist())
    return float(np.mean(losses)) if losses else 0.0, priorities


def fit_task_alignment(density: np.ndarray, config: dict) -> dict:
    normalized = row_normalize(density)
    kernel = np.clip(normalized @ normalized.T, -1.0, 1.0)
    receiver_count = len(normalized)
    candidates = []
    maximum_clusters = min(int(config["maximum_clusters"]), receiver_count)
    for cluster_count in range(1, maximum_clusters + 1):
        if cluster_count == 1:
            labels = np.zeros(receiver_count, dtype=int)
        else:
            model = AgglomerativeClustering(
                n_clusters=cluster_count,
                metric="precomputed",
                linkage="average",
            )
            labels = model.fit_predict(np.maximum(0.0, 1.0 - kernel))
        loss, priorities = clustering_loss(normalized, labels)
        penalty = float(config["representation_penalty"]) * (cluster_count - 1) / max(receiver_count - 1, 1)
        minimum_alignment = min(
            float(normalized[index] @ priorities[int(labels[index])])
            for index in range(receiver_count)
        )
        objective = loss + penalty
        candidates.append(
            {
                "cluster_count": cluster_count,
                "labels": labels,
                "priorities": priorities,
                "mean_misalignment": loss,
                "minimum_alignment": minimum_alignment,
                "objective": objective,
            }
        )
    feasible = [
        item
        for item in candidates
        if item["minimum_alignment"] >= float(config["minimum_task_alignment"])
    ]
    selected = min(feasible or candidates, key=lambda item: (item["objective"], item["cluster_count"]))
    return {"kernel": kernel, "normalized": normalized, "selected": selected, "candidates": candidates}


def compress_to_budget(tokenizer, token_ids: np.ndarray, costs: np.ndarray, priority: np.ndarray, removal: int):
    total_cost = float(costs.sum())
    budget = total_cost * (1.0 - int(removal) / 100.0)
    ranked = sorted(range(len(token_ids)), key=lambda index: (-float(priority[index]), index))
    kept = []
    used = 0.0
    for index in ranked:
        cost = float(costs[index])
        if used + cost <= budget or not kept:
            kept.append(index)
            used += cost
    kept_set = set(kept)
    ordered_ids = [int(token_id) for index, token_id in enumerate(token_ids) if index in kept_set]
    compressed = tokenizer.decode(ordered_ids, skip_special_tokens=True)
    transmitted_bits = len(compressed.encode("utf-8", errors="replace")) * 8
    return compressed, len(ordered_ids), transmitted_bits


def append_csv(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_contexts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["context_id"])
    return set(frame["context_id"].astype(str))


def evaluate_context(
    cached: dict,
    tokenizer,
    qa,
    similarity_model,
    config: dict,
    output_dir: Path,
) -> None:
    context_id = str(cached["context_id"].item())
    receiver_ids = cached["receiver_ids"].astype(int)
    questions = cached["questions"].astype(str).tolist()
    answers = [parse_answers(value) for value in cached["true_answers"].tolist()]
    token_ids = cached["token_ids"].astype(int)
    costs = cached["token_cost_bits"].astype(float)
    density = cached["density"].astype(float)
    alignment = fit_task_alignment(density, config)
    normalized = alignment["normalized"]
    selected = alignment["selected"]

    kernel_rows = []
    for left in range(len(receiver_ids)):
        for right in range(len(receiver_ids)):
            kernel_rows.append(
                {
                    "context_id": context_id,
                    "receiver_i": int(receiver_ids[left]),
                    "receiver_j": int(receiver_ids[right]),
                    "alignment": float(alignment["kernel"][left, right]),
                }
            )
    append_csv(output_dir / "receiver_kernel_matrix.csv", kernel_rows)

    candidate_rows = []
    for candidate in alignment["candidates"]:
        candidate_rows.append(
            {
                "context_id": context_id,
                "receiver_count": len(receiver_ids),
                "cluster_count": candidate["cluster_count"],
                "mean_misalignment": candidate["mean_misalignment"],
                "minimum_alignment": candidate["minimum_alignment"],
                "representation_penalty": float(config["representation_penalty"]),
                "objective": candidate["objective"],
                "selected": candidate is selected,
            }
        )
    append_csv(output_dir / "cluster_objectives.csv", candidate_rows)

    token_rows = []
    for receiver_index, receiver_id in enumerate(receiver_ids):
        for token_index, token_id in enumerate(token_ids):
            token_rows.append(
                {
                    "context_id": context_id,
                    "receiver_id": int(receiver_id),
                    "question": questions[receiver_index],
                    "token_position": token_index,
                    "token_id": int(token_id),
                    "token_text": str(cached["token_text"][token_index]),
                    "token_cost_bits": float(costs[token_index]),
                    "relevance": float(cached["relevance"][receiver_index, token_index]),
                    "semantic_density": float(density[receiver_index, token_index]),
                    "selected_cluster": int(selected["labels"][receiver_index]),
                }
            )
    append_csv(output_dir / "token_relevance_and_density.csv", token_rows)

    strategy_labels = {
        "global_alignment": np.zeros(len(receiver_ids), dtype=int),
        "task_alignment": selected["labels"].astype(int),
        "individual_tasks": np.arange(len(receiver_ids), dtype=int),
    }
    result_rows = []
    assignment_rows = []
    for strategy, labels in strategy_labels.items():
        priority_objective = (
            str(config.get("global_sa_misalignment", "cosine"))
            if strategy == "global_alignment"
            else "cosine"
        )
        _loss, priorities = clustering_loss(
            normalized,
            labels,
            priority_objective=priority_objective,
        )
        for receiver_index, receiver_id in enumerate(receiver_ids):
            assignment_rows.append(
                {
                    "context_id": context_id,
                    "strategy": strategy,
                    "receiver_id": int(receiver_id),
                    "question": questions[receiver_index],
                    "cluster_id": int(labels[receiver_index]),
                }
            )
        for removal in config["removal_percentages"]:
            compressed_by_cluster = {}
            for cluster, priority in priorities.items():
                compressed_by_cluster[cluster] = compress_to_budget(
                    tokenizer, token_ids, costs, priority, int(removal)
                )
            contexts = [compressed_by_cluster[int(labels[index])][0] for index in range(len(receiver_ids))]
            outputs = qa_predictions(qa, questions, contexts, int(config["qa_batch_size"]))
            total_bits = sum(item[2] for item in compressed_by_cluster.values())
            for receiver_index, output in enumerate(outputs):
                prediction = str(output.get("answer", ""))
                accuracy, exact_match, f1, similarity = evaluate_prediction(
                    prediction, answers[receiver_index], similarity_model
                )
                cluster = int(labels[receiver_index])
                compressed, kept_count, cluster_bits = compressed_by_cluster[cluster]
                result_rows.append(
                    {
                        "context_id": context_id,
                        "receiver_id": int(receiver_ids[receiver_index]),
                        "question": questions[receiver_index],
                        "strategy": strategy,
                        "target_removal_percent": int(removal),
                        "cluster_id": cluster,
                        "representation_count": len(compressed_by_cluster),
                        "original_token_count": len(token_ids),
                        "kept_token_count": kept_count,
                        "actual_token_reduction": 1.0 - kept_count / len(token_ids),
                        "cluster_payload_bits": cluster_bits,
                        "total_transmitted_bits": total_bits,
                        "prediction": prediction,
                        "accuracy": accuracy,
                        "exact_match": float(exact_match),
                        "f1": f1,
                        "semantic_similarity": similarity,
                        "compressed_context": compressed,
                    }
                )
    append_csv(output_dir / "cluster_assignments.csv", assignment_rows)
    append_csv(output_dir / "receiver_results.csv", result_rows)


def summarize(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_path = output_dir / "receiver_results.csv"
    if not result_path.exists():
        raise FileNotFoundError(f"No receiver results found: {result_path}")
    rows = pd.read_csv(result_path)
    context_summary = (
        rows.groupby(["context_id", "strategy", "target_removal_percent"])
        .agg(
            receivers=("receiver_id", "nunique"),
            representations=("representation_count", "first"),
            actual_token_reduction=("actual_token_reduction", "mean"),
            total_transmitted_bits=("total_transmitted_bits", "first"),
            mean_accuracy=("accuracy", "mean"),
            weakest_receiver_accuracy=("accuracy", "min"),
            mean_exact_match=("exact_match", "mean"),
            mean_f1=("f1", "mean"),
            mean_semantic_similarity=("semantic_similarity", "mean"),
        )
        .reset_index()
    )
    summary = (
        context_summary.groupby(["strategy", "target_removal_percent"])
        .agg(
            contexts=("context_id", "nunique"),
            mean_receivers=("receivers", "mean"),
            mean_representations=("representations", "mean"),
            actual_token_reduction=("actual_token_reduction", "mean"),
            mean_total_transmitted_bits=("total_transmitted_bits", "mean"),
            mean_accuracy=("mean_accuracy", "mean"),
            weakest_receiver_accuracy=("weakest_receiver_accuracy", "mean"),
            mean_exact_match=("mean_exact_match", "mean"),
            mean_f1=("mean_f1", "mean"),
            mean_semantic_similarity=("mean_semantic_similarity", "mean"),
        )
        .reset_index()
    )
    context_summary.to_csv(output_dir / "context_summary.csv", index=False)
    summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    return context_summary, summary


def summarize_all_questions(output_dir: Path) -> pd.DataFrame:
    rows = pd.read_csv(output_dir / "receiver_results.csv")
    summary = (
        rows.groupby(["strategy", "target_removal_percent"])
        .agg(
            receiver_questions=("receiver_id", "count"),
            contexts=("context_id", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_exact_match=("exact_match", "mean"),
            mean_f1=("f1", "mean"),
            mean_semantic_similarity=("semantic_similarity", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "all_questions_strategy_summary.csv", index=False)
    return summary


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    styles = {
        "global_alignment": ("#2563eb", "--", "o"),
        "task_alignment": ("#dc2626", "-", "o"),
        "individual_tasks": ("#16a34a", ":", "s"),
    }
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for strategy, label in STRATEGIES.items():
        group = summary[summary["strategy"] == strategy].sort_values("target_removal_percent")
        if group.empty:
            continue
        color, linestyle, marker = styles[strategy]
        ax.plot(
            group["target_removal_percent"],
            group[metric],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.4,
            markersize=6,
        )
    ax.set_xlabel("Target payload reduction (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["target_removal_percent"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_report(config: dict, output_dir: Path, summary: pd.DataFrame) -> None:
    global_objective = str(config.get("global_sa_misalignment", "cosine"))
    overall = (
        summary.groupby("strategy")
        .agg(
            mean_accuracy=("mean_accuracy", "mean"),
            weakest_receiver_accuracy=("weakest_receiver_accuracy", "mean"),
            mean_f1=("mean_f1", "mean"),
            mean_semantic_similarity=("mean_semantic_similarity", "mean"),
            mean_representations=("mean_representations", "mean"),
            mean_total_transmitted_bits=("mean_total_transmitted_bits", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Paper-Style Multi-Receiver Task Alignment Results",
        "",
        "## Setup",
        "",
        f"- Input: `{resolve(config['input_csv'])}`",
        "- One context is shared by multiple QA receivers.",
        "- P is instantiated as the L2 norm of a receiver-specific QA distortion gradient.",
        "- Semantic density is relevance per token bit cost (paper Eq. 5).",
        "- Receiver alignment uses a cosine kernel (paper Eq. 20-21).",
        f"- Global SA co-priority objective: `{global_objective}` misalignment (paper Eq. 9).",
        "- Cluster co-priorities use cosine misalignment; receiver grouping uses paper Eq. 20-21.",
        "",
        "## Important Implementation Boundary",
        "",
        "The paper defines the objective but does not prescribe one concrete importance estimator, clustering algorithm, "
        "or cluster-count rule. This experiment uses the paper's distortion-gradient option for P, agglomerative "
        "clustering, and alignment loss plus a penalty for additional representations.",
        "",
        "## Overall Results",
        "",
        "| Strategy | Accuracy | Weakest receiver | F1 | Similarity | Representations | Bits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {STRATEGIES.get(row.strategy, row.strategy)} | {row.mean_accuracy:.4f} | "
            f"{row.weakest_receiver_accuracy:.4f} | {row.mean_f1:.4f} | "
            f"{row.mean_semantic_similarity:.4f} | {row.mean_representations:.2f} | "
            f"{row.mean_total_transmitted_bits:.1f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_outputs(config: dict, output_dir: Path) -> None:
    _context_summary, summary = summarize(output_dir)
    all_questions_summary = summarize_all_questions(output_dir)
    plots = output_dir / "plots"
    plot_metric(summary, "mean_accuracy", "Mean receiver accuracy", plots / "mean_accuracy.svg")
    plot_metric(
        summary,
        "weakest_receiver_accuracy",
        "Weakest-receiver accuracy",
        plots / "weakest_receiver_accuracy.svg",
    )
    plot_metric(summary, "mean_f1", "Mean receiver F1", plots / "mean_f1.svg")
    plot_metric(
        summary,
        "mean_semantic_similarity",
        "Mean semantic similarity",
        plots / "semantic_similarity.svg",
    )
    plot_metric(
        summary,
        "mean_total_transmitted_bits",
        "Mean total transmitted bits per context",
        plots / "transmitted_bits.svg",
    )
    plot_metric(
        summary,
        "mean_representations",
        "Mean transmitted representations per context",
        plots / "representation_count.svg",
    )
    plot_metric(
        all_questions_summary,
        "mean_accuracy",
        "Accuracy across all receiver questions",
        plots / "mean_accuracy_all_questions.svg",
    )
    write_report(config, output_dir, summary)


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.plot_only:
        make_outputs(config, output_dir)
        print(f"Plots and report written to: {output_dir}")
        return

    receivers = load_receivers(config, args.max_contexts)
    context_sizes = receivers.groupby("context_id").size()
    estimated_relevance_calls = len(receivers)
    print(
        json.dumps(
            {
                "samples_receivers": len(receivers),
                "contexts": int(receivers["context_id"].nunique()),
                "mean_receivers_per_context": float(context_sizes.mean()),
                "estimated_importance_forward_backward_passes": estimated_relevance_calls,
                "removal_percentages": config["removal_percentages"],
                "strategies": list(STRATEGIES),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    tokenizer, qa_model, qa = make_qa_pipeline(config["qa_model"])
    similarity_model = SentenceTransformer(config["similarity_model"], local_files_only=True)
    receiver_results_path = output_dir / "receiver_results.csv"
    done = completed_contexts(receiver_results_path)
    groups = list(receivers.groupby("context_id", sort=True))
    for position, (context_id, group) in enumerate(groups, start=1):
        if str(context_id) in done:
            print(f"Context {position}/{len(groups)} already complete: {context_id}")
            continue
        print(f"Context {position}/{len(groups)}: {context_id}, receivers={len(group)}")
        cached = compute_relevance_for_context(group, tokenizer, qa_model, config, output_dir)
        evaluate_context(cached, tokenizer, qa, similarity_model, config, output_dir)
        done.add(str(context_id))

    make_outputs(config, output_dir)
    print(f"Experiment complete: {output_dir}")


if __name__ == "__main__":
    main()
