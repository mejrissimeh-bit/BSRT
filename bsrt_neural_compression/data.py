from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from .common import digest, read_json, resolve, write_json


def final_records(config):
    path = resolve(config["final_archive"]) / "standard_methods_sample_results.csv"
    samples = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            samples.setdefault(row["squad_id"], {
                "id": row["squad_id"], "context": row["original_context"],
                "question": row["question"], "answers": {"text": read_answers(row["true_answers"]), "answer_start": []},
                "dataset_index": int(row["dataset_index"]),
            })
    if len(samples) != 500:
        raise ValueError(f"Final archive must contain exactly 500 QA pairs, got {len(samples)}")
    return list(samples.values())


def read_answers(text):
    import ast
    return ast.literal_eval(text)


def split_manifest(config):
    path = resolve(config["output_dir"]) / "splits.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["config_fingerprint"] != config["config_fingerprint"]:
            raise ValueError("Configuration changed: use a new output_dir to preserve this experiment.")
        return manifest
    source = resolve(config["dataset_path"])
    if not source.exists():
        raise FileNotFoundError(f"Download the official SQuAD 2.0 training JSON to {source} (see README).")
    dataset = read_json(source)
    final = final_records(config)
    forbidden_contexts = {digest(row["context"]) for row in final}
    forbidden_ids = {row["id"] for row in final}
    tokenizer = AutoTokenizer.from_pretrained(config["qa_model"], local_files_only=config["local_files_only"])
    articles = defaultdict(list)
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_id = digest(context)
            if context_id in forbidden_contexts or any(q["id"] in forbidden_ids for q in paragraph["qas"]):
                continue
            questions = [{"id": q["id"], "question": q["question"], "answers": {
                "text": [a["text"] for a in q["answers"]],
                "answer_start": [a["answer_start"] for a in q["answers"]],
            }} for q in paragraph["qas"]]
            if len(questions) < 2 or not any(q["answers"]["text"] for q in questions):
                continue
            articles[article["title"]].append({"context_id": context_id, "context": context, "article": article["title"], "questions": questions})
    rng = np.random.default_rng(config["seed"])
    titles = sorted(articles)
    rng.shuffle(titles)
    n = len(titles)
    partitions = {"train": titles[:int(.8*n)], "validation": titles[int(.8*n):int(.9*n)], "untouched": titles[int(.9*n):]}
    selected = {}
    used_test_contexts = set()
    for previous in config.get("exclude_untouched_from", []):
        old_manifest = read_json(resolve(previous))
        used_test_contexts.update(row["context_id"] for row in old_manifest["untouched"])
    for name, article_titles in partitions.items():
        # Interleave articles so a pilot is not drawn entirely from one topic.
        queues = [list(articles[title]) for title in article_titles]
        for queue in queues:
            rng.shuffle(queue)
        candidates = [queue[i] for i in range(max(map(len, queues))) for queue in queues if i < len(queue)]
        selected[name] = []
        seen = set()
        limit = config[f"{name}_contexts"]
        for row in candidates:
            if name == "untouched" and row["context_id"] in used_test_contexts:
                continue
            if row["context_id"] in seen:
                continue
            ids = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
            if len(ids) > config["max_context_tokens"] or len(ids) < 24:
                continue
            selected[name].append(row)
            seen.add(row["context_id"])
            if len(selected[name]) >= limit:
                break
        if len(selected[name]) != limit:
            raise ValueError(f"Insufficient eligible {name} contexts")
    sets = [{r["context_id"] for r in selected[name]} for name in selected]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Duplicate contexts across partitions")
    manifest = {"config_fingerprint": config["config_fingerprint"], "dataset": "SQuAD 2.0 train", "dataset_sha256": digest(source.read_text(encoding="utf-8")),
                "final_excluded_contexts": len(forbidden_contexts), "final_excluded_questions": 500,
                "final_subset_sha256": digest(final), **selected}
    write_json(path, manifest)
    return manifest


def training_questions(record, config):
    questions = list(record["questions"])
    rng = np.random.default_rng(int(record["context_id"][:8], 16) + config["seed"])
    rng.shuffle(questions)
    # Include answerable and unanswerable supervision where available.
    positive = [q for q in questions if q["answers"]["text"]]
    negative = [q for q in questions if not q["answers"]["text"]]
    ordered = []
    for i in range(max(len(positive), len(negative))):
        if i < len(positive): ordered.append(positive[i])
        if i < len(negative): ordered.append(negative[i])
    return ordered[:config["max_train_questions"]]
