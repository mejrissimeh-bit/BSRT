from io import StringIO
import os
import re
import string

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


def get_int_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def get_sweep_percentages():
    sweep_min = get_int_env("SWEEP_MIN", 0)
    sweep_max = get_int_env("SWEEP_MAX", 30)
    sweep_step = get_int_env("SWEEP_STEP", 5)

    if sweep_step <= 0:
        raise ValueError("SWEEP_STEP must be greater than 0")
    if sweep_min > sweep_max:
        raise ValueError("SWEEP_MIN must be less than or equal to SWEEP_MAX")

    return list(range(sweep_min, sweep_max + 1, sweep_step))


def normalize_answer(answer):
    answer = answer.lower()
    answer = "".join(char for char in answer if char not in string.punctuation)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = " ".join(answer.split())
    return answer


def exact_match_score(predicted_answer, true_answer):
    return normalize_answer(predicted_answer) == normalize_answer(true_answer)


def f1_score(predicted_answer, true_answer):
    predicted_tokens = normalize_answer(predicted_answer).split()
    true_tokens = normalize_answer(true_answer).split()

    common_tokens = set(predicted_tokens) & set(true_tokens)
    common_count = sum(
        min(predicted_tokens.count(token), true_tokens.count(token))
        for token in common_tokens
    )

    if len(predicted_tokens) == 0 or len(true_tokens) == 0:
        return int(predicted_tokens == true_tokens)

    if common_count == 0:
        return 0

    precision = common_count / len(predicted_tokens)
    recall = common_count / len(true_tokens)
    return 2 * precision * recall / (precision + recall)


def is_correct_answer(
    predicted_answer,
    true_answers,
    f1,
    semantic_similarity,
    f1_threshold=0.5,
    semantic_threshold=0.75,
):
    normalized_prediction = normalize_answer(predicted_answer)

    for true_answer in true_answers:
        normalized_true_answer = normalize_answer(true_answer)

        if normalized_prediction == normalized_true_answer:
            return True

        if normalized_true_answer in normalized_prediction:
            return True

    return f1 >= f1_threshold or semantic_similarity >= semantic_threshold


def remove_low_importance_tokens(context, tokenizer, model, removal_ratio=0.2):
    encoded_context = tokenizer(context, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    input_ids = encoded_context["input_ids"].to(device)
    token_ids = input_ids[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    special_token_ids = set(tokenizer.all_special_ids)

    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(input_ids)[0]
        importance_scores = torch.norm(token_embeddings, p=2, dim=1)

    removable_indices = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id not in special_token_ids
    ]
    remove_count = int(len(removable_indices) * removal_ratio)

    indices_by_importance = sorted(
        removable_indices,
        key=lambda index: importance_scores[index].item(),
    )
    removed_indices = set(indices_by_importance[:remove_count])
    token_importances = [
        (tokens[index], importance_scores[index].item())
        for index in removable_indices
    ]
    removed_tokens = [
        (tokens[index], importance_scores[index].item())
        for index in indices_by_importance[:remove_count]
    ]

    if remove_count == 0:
        return context, token_importances, removed_tokens

    kept_token_ids = [
        token_id
        for index, token_id in enumerate(token_ids)
        if index not in removed_indices and token_id not in special_token_ids
    ]

    modified_context = tokenizer.decode(
        kept_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return modified_context, token_importances, removed_tokens


def evaluate_prediction(predicted_answer, true_answers, similarity_model):
    exact_match = max(
        exact_match_score(predicted_answer, true_answer)
        for true_answer in true_answers
    )
    f1 = max(
        f1_score(predicted_answer, true_answer)
        for true_answer in true_answers
    )
    predicted_embedding = similarity_model.encode(predicted_answer)
    true_embeddings = similarity_model.encode(true_answers)
    similarity = util.cos_sim(predicted_embedding, true_embeddings).max().item()
    accuracy = int(is_correct_answer(predicted_answer, true_answers, f1, similarity))

    return accuracy, exact_match, f1, similarity


def empty_totals():
    return {
        "correct": 0,
        "exact_match": 0,
        "f1": 0,
        "similarity": 0,
    }


def add_metrics(totals, metrics):
    accuracy, exact_match, f1, similarity = metrics
    totals["correct"] += accuracy
    totals["exact_match"] += exact_match
    totals["f1"] += f1
    totals["similarity"] += similarity


def average_metrics(totals, sample_count):
    if sample_count == 0:
        return {
            "accuracy": 0,
            "exact_match": 0,
            "f1": 0,
            "similarity": 0,
        }

    return {
        "accuracy": totals["correct"] / sample_count,
        "exact_match": totals["exact_match"] / sample_count,
        "f1": totals["f1"] / sample_count,
        "similarity": totals["similarity"] / sample_count,
    }


def write_percentage_report(
    file_name,
    percent,
    total_samples,
    selected_samples,
    original_metrics,
    removal_totals,
    removal_metrics,
    details_text,
):
    with open(file_name, "w", encoding="utf-8") as report:
        print("L2 Token Removal Degradation Report", file=report)
        print("=" * 80, file=report)
        print("Removed token percentage:", f"{percent}%", file=report)
        print("Total samples checked:", total_samples, file=report)
        print("Samples used for degradation evaluation:", selected_samples, file=report)
        print(file=report)
        print("Original metrics on selected samples", file=report)
        print("Accuracy:", original_metrics["accuracy"], file=report)
        print("Exact Match:", original_metrics["exact_match"], file=report)
        print("F1 score:", original_metrics["f1"], file=report)
        print("Average semantic similarity:", original_metrics["similarity"], file=report)
        print(file=report)
        print("Metrics after token removal", file=report)
        print("Correct answers after removal:", removal_totals["correct"], file=report)
        print("Accuracy:", removal_metrics["accuracy"], file=report)
        print("Exact Match:", removal_metrics["exact_match"], file=report)
        print("F1 score:", removal_metrics["f1"], file=report)
        print("Average semantic similarity:", removal_metrics["similarity"], file=report)
        print(file=report)
        print("Performance drop", file=report)
        print(
            "Accuracy drop:",
            original_metrics["accuracy"] - removal_metrics["accuracy"],
            file=report,
        )
        print(
            "Exact Match drop:",
            original_metrics["exact_match"] - removal_metrics["exact_match"],
            file=report,
        )
        print("F1 drop:", original_metrics["f1"] - removal_metrics["f1"], file=report)
        print(
            "Semantic similarity drop:",
            original_metrics["similarity"] - removal_metrics["similarity"],
            file=report,
        )
        print("=" * 80, file=report)
        print(file=report)
        print("Detailed samples", file=report)
        print("=" * 80, file=report)
        print(details_text, file=report)


def main():
    total_samples = get_int_env("TOTAL_SAMPLES", 500)
    removal_percentages = get_sweep_percentages()
    summary_file = "l2_degradation_summary.txt"
    dataset = load_dataset("squad", split=f"validation[:{total_samples}]")

    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/bert-base-cased-squad2",
        tokenizer="deepset/bert-base-cased-squad2",
    )
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    original_totals = empty_totals()
    removal_totals = {
        percent: empty_totals()
        for percent in removal_percentages
    }
    report_details = {
        percent: StringIO()
        for percent in removal_percentages
    }
    selected_samples = 0

    for sample_number, sample in enumerate(dataset, start=1):
        prediction = qa_pipeline(
            question=sample["question"],
            context=sample["context"],
        )

        predicted_answer = prediction["answer"]
        true_answers = sample["answers"]["text"]
        original_metrics = evaluate_prediction(
            predicted_answer,
            true_answers,
            similarity_model,
        )
        original_accuracy, original_exact_match, original_f1, original_similarity = (
            original_metrics
        )

        selected_samples += 1
        add_metrics(original_totals, original_metrics)

        for percent in removal_percentages:
            removal_ratio = percent / 100
            modified_context, token_importances, removed_tokens = (
                remove_low_importance_tokens(
                    sample["context"],
                    qa_pipeline.tokenizer,
                    qa_pipeline.model,
                    removal_ratio,
                )
            )
            removal_prediction = qa_pipeline(
                question=sample["question"],
                context=modified_context,
            )
            removal_answer = removal_prediction["answer"]
            removal_metrics = evaluate_prediction(
                removal_answer,
                true_answers,
                similarity_model,
            )
            removal_accuracy, removal_exact_match, removal_f1, removal_similarity = (
                removal_metrics
            )
            add_metrics(removal_totals[percent], removal_metrics)

            details = report_details[percent]
            print("Sample:", sample_number, file=details)
            print("Question:", sample["question"], file=details)
            print("True answers:", true_answers, file=details)
            print("Original predicted answer:", predicted_answer, file=details)
            print("Original Exact Match:", original_exact_match, file=details)
            print("Original F1:", original_f1, file=details)
            print("Original semantic similarity:", original_similarity, file=details)
            print("Predicted answer after removal:", removal_answer, file=details)
            print("Exact Match after removal:", removal_exact_match, file=details)
            print("F1 after removal:", removal_f1, file=details)
            print("Semantic similarity after removal:", removal_similarity, file=details)
            print(
                "Correct after removal:",
                bool(removal_accuracy),
                file=details,
            )
            print(file=details)
            print("L2 token priorities", file=details)
            for token, score in token_importances:
                print(f"{token}: {score:.4f}", file=details)
            print(file=details)
            print("Removed tokens", file=details)
            for token, score in removed_tokens:
                print(f"{token}: {score:.4f}", file=details)
            print(file=details)
            print("Modified context:", file=details)
            print(modified_context, file=details)
            print("-" * 80, file=details)
            print(file=details)

    original_metrics = average_metrics(original_totals, selected_samples)
    removal_metrics_by_percent = {
        percent: average_metrics(removal_totals[percent], selected_samples)
        for percent in removal_percentages
    }

    report_files = []
    for percent in removal_percentages:
        file_name = f"l2_degradation_{percent:02d}_percent.txt"
        report_files.append(file_name)
        details_text = report_details[percent].getvalue()
        report_details[percent].close()
        write_percentage_report(
            file_name,
            percent,
            total_samples,
            selected_samples,
            original_metrics,
            removal_totals[percent],
            removal_metrics_by_percent[percent],
            details_text,
        )

    with open(summary_file, "w", encoding="utf-8") as report:
        print("L2 Token Removal Sweep Summary", file=report)
        print("=" * 80, file=report)
        print("Total samples checked:", total_samples, file=report)
        print("Selected originally correct samples:", selected_samples, file=report)
        print(file=report)
        print("Original metrics on selected samples", file=report)
        print("Accuracy:", original_metrics["accuracy"], file=report)
        print("Exact Match:", original_metrics["exact_match"], file=report)
        print("F1 score:", original_metrics["f1"], file=report)
        print("Average semantic similarity:", original_metrics["similarity"], file=report)
        print(file=report)
        print("Removal sweep", file=report)
        print("Percent | Accuracy | Exact Match | F1 | Semantic Similarity", file=report)
        for percent in removal_percentages:
            metrics = removal_metrics_by_percent[percent]
            print(
                f"{percent}% | {metrics['accuracy']} | "
                f"{metrics['exact_match']} | {metrics['f1']} | "
                f"{metrics['similarity']}",
                file=report,
            )

    print("Total samples checked:", total_samples)
    print("Selected originally correct samples:", selected_samples)
    print("Summary report:", summary_file)
    print("Individual percentage reports:")
    for file_name in report_files:
        print(file_name)


if __name__ == "__main__":
    main()
