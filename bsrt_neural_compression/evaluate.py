from __future__ import annotations

import csv
import inspect
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .common import digest, read_json, resolve, write_json
from .data import final_records, split_manifest
from .model import BSRT, ContextFeatures, RetentionNetwork, render_context
from .reward import QAReward


def old_policy(qa, config):
    import joblib
    return joblib.load(resolve(config["old_policy_checkpoint"]))


def standard_text(method, context, percent, qa, old=None, seed=42):
    if method == "original":
        return context
    if method == "random":
        from random_removal import remove_random_tokens
        return remove_random_tokens(context, qa.tokenizer, removal_ratio=percent/100, seed=seed)[0]
    if method == "l2":
        from main import remove_low_importance_tokens
        return remove_low_importance_tokens(context, qa.tokenizer, qa.model, percent/100)[0]
    if method == "attention":
        from attention_importance import remove_low_attention_tokens
        return remove_low_attention_tokens(context, qa.tokenizer, qa.model, percent/100)[0]
    if method == "learned_token_policy":
        from learned_token_policy.train_and_evaluate_token_policy import compress_with_policy
        return compress_with_policy(context, percent, qa.tokenizer, qa.model, old["idf"], old["model"], old["feature_columns"])
    raise ValueError(method)


def append_results(path, rows):
    if not rows:
        return
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summary_tables(rows, path):
    frame = pd.DataFrame(rows)
    metrics = [column for column in ["accuracy", "f1", "semantic_similarity", "actual_token_reduction", "compression_seconds"] if column in frame and frame[column].notna().any()]
    result = frame.groupby(["method", "removal_percent"])[metrics].mean().reset_index()
    result.to_csv(path / "comparison_summary.csv", index=False)
    for answerable, subset in frame.groupby("answerable"):
        subset.groupby(["method", "removal_percent"])[metrics].mean().to_csv(path / ("answerable_summary.csv" if answerable else "unanswerable_summary.csv"))
    return result


def evaluate_development(config, partition="validation", qa=None, checkpoint_path=None, include_baselines=True):
    if partition not in {"validation", "untouched"}:
        raise ValueError("Only validation and untouched development partitions allowed")
    manifest = split_manifest(config)
    output = resolve(config["output_dir"])
    checkpoint_path = checkpoint_path or output / "checkpoints" / "best.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved["config"]["config_fingerprint"] != config["config_fingerprint"]:
        raise ValueError("Checkpoint and configuration do not match")
    checkpoint_hash = digest(checkpoint_path.read_bytes())
    destination = output / "evaluation" / partition
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"checkpoint_hash": checkpoint_hash, "partition": partition, "config_fingerprint": config["config_fingerprint"], "baselines": include_baselines}
    meta_path = destination / "provenance.json"
    if meta_path.exists() and read_json(meta_path) != metadata:
        raise ValueError("Evaluation checkpoint changed; use a separate output directory")
    write_json(meta_path, metadata)
    owns_qa = qa is None
    qa = qa or QAReward(config)
    features = ContextFeatures(config, qa.model, qa.tokenizer)
    network = RetentionNetwork(**config["model"])
    network.load_state_dict(saved["state_dict"])
    policy = BSRT(network.to(qa.device), features)
    old = old_policy(qa, config) if include_baselines else None
    path = destination / "sample_results.csv"
    completed = set()
    if path.exists():
        previous = pd.read_csv(path)
        completed = set(zip(previous.context_id, previous.removal_percent, previous.method))
    methods = ["original", "random", "l2", "attention", "learned_token_policy", "bsrt"] if include_baselines else ["bsrt"]
    for i, record in enumerate(manifest[partition], 1):
        for percent in config["removal_percentages"]:
            for method in methods:
                key = record["context_id"], percent, method
                if key in completed:
                    continue
                start = time.perf_counter()
                compressed = policy.compress(record["context"], percent/100) if method == "bsrt" else standard_text(method, record["context"], percent, qa, old)
                seconds = time.perf_counter()-start
                token_count = len(qa.tokenizer(record["context"], add_special_tokens=False)["input_ids"])
                kept = len(qa.tokenizer(compressed, add_special_tokens=False)["input_ids"])
                metrics = qa.metrics(compressed, record["questions"], allow_no_answer=True, semantic=True)
                rows = [{**value, "context_id": record["context_id"], "method": method, "removal_percent": percent,
                         "actual_token_reduction": 1-kept/token_count, "compression_seconds": seconds} for value in metrics]
                append_results(path, rows)
            print(f"BSRT {partition} context {i}/{len(manifest[partition])} removal={percent}%", flush=True)
    rows = pd.read_csv(path).to_dict("records")
    result = summary_tables(rows, destination)
    # Benchmark encoder computation separately: the training cache would
    # otherwise make the neural compressor appear artificially inexpensive.
    class FreshFeatures:
        def get(self, context):
            return features.get(context, use_cache=False)
    cold_policy = BSRT(network, FreshFeatures())
    timings = []
    if "use_cache" in inspect.signature(features.get).parameters:
        for record in manifest[partition][:min(4, len(manifest[partition]))]:
            start = time.perf_counter()
            cold_policy.compress(record["context"], config["removal_percentages"][0]/100)
            timings.append(time.perf_counter()-start)
    write_json(destination / "inference_timing.json", {
        "cold_encoder_mean_seconds": float(np.mean(timings)) if timings else None,
        "measured_contexts": len(timings), "model_loading_included": False,
        "note": "Compression seconds in the sample table use cached BSRT encoder features; this cold measurement includes BERT encoding and selection."
    })
    if owns_qa:
        qa.close()
    return result


def final_evaluation(config):
    """Evaluate the frozen checkpoint on the exact final 500; no mask search."""
    output = resolve(config["output_dir"])
    levels = config.get("evaluation_percentages", config["removal_percentages"])
    checkpoint_path = output / "checkpoints" / "best.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved["config"]["config_fingerprint"] != config["config_fingerprint"]:
        raise ValueError("Checkpoint and configuration do not match")
    if not (output / "training_summary.json").exists():
        raise ValueError("Complete training and freeze the checkpoint before final evaluation")
    manifest = split_manifest(config)
    final = final_records(config)
    forbidden = {digest(row["context"]) for row in final}
    assert all(row["context_id"] not in forbidden for split in ["train", "validation", "untouched"] for row in manifest[split])
    destination = output / "evaluation" / "final_500"
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / "frozen_checkpoint.json"
    metadata = {"checkpoint_sha256": digest(checkpoint_path.read_bytes()),
                "config_fingerprint": config["config_fingerprint"], "sample_sha256": digest(final),
                "prior_test_reuse": "The thesis 500 have been inspected repeatedly in previous research; they are a fixed comparison set, not an untouched discovery test."}
    if lock_path.exists() and read_json(lock_path) != metadata:
        raise ValueError("Final evaluation is locked to its checkpoint. Do not tune on the 500.")
    write_json(lock_path, metadata)
    qa = QAReward(config)
    network = RetentionNetwork(**config["model"])
    network.load_state_dict(saved["state_dict"])
    policy = BSRT(network.to(qa.device), ContextFeatures(config, qa.model, qa.tokenizer))
    groups = defaultdict(list)
    for row in final:
        groups[row["context"]].append(row)
    path = destination / "bsrt_sample_results.csv"
    completed = set()
    if path.exists():
        previous = pd.read_csv(path)
        completed = set(zip(previous.question_id, previous.removal_percent))
    for i, (context, questions) in enumerate(groups.items(), 1):
        for percent in levels:
            missing = [row for row in questions if (row["id"], percent) not in completed]
            if not missing:
                continue
            start = time.perf_counter()
            mask, feature = policy.select(context, percent/100)
            compressed = render_context(context, feature["offsets"], mask)
            seconds = time.perf_counter()-start
            kept = len(qa.tokenizer(compressed, add_special_tokens=False)["input_ids"])
            metrics = qa.metrics(compressed, missing, allow_no_answer=False, semantic=True)
            lookup = {row["id"]: row for row in missing}
            rows = [{**value, "dataset_index": lookup[value["question_id"]]["dataset_index"],
                     "context_id": digest(context), "method": "bsrt", "removal_percent": percent,
                     "original_token_count": len(feature["input_ids"]), "kept_mask_tokens": int(mask.sum()),
                     "compressed_token_count": kept, "actual_token_reduction": 1-kept/len(feature["input_ids"]),
                     "compression_seconds": seconds, "compressed_context": compressed} for value in metrics]
            append_results(path, rows)
            print(f"BSRT final context {i}/{len(groups)} removal={percent}% questions={len(missing)}", flush=True)
    baseline = pd.read_csv(resolve(config["final_archive"]) / "all_strategy_sample_results.csv")
    baseline = baseline[baseline.removal_percent.isin(levels)].copy()
    baseline["context_id"] = baseline.original_context.map(digest)
    baseline["question_id"] = baseline.squad_id
    baseline["answerable"] = True
    new = pd.read_csv(path)
    if len(new) != len(final)*len(levels):
        raise ValueError("Incomplete final evaluation")
    combined = pd.concat([baseline, new], ignore_index=True)
    original = baseline.drop_duplicates(["question_id", "removal_percent"]).copy()
    original["method"] = "original"
    for metric in ["accuracy", "exact_match", "f1", "semantic_similarity"]:
        original[metric] = original["original_" + metric]
    original["actual_token_reduction"] = 0.0
    original["compression_seconds"] = 0.0
    # Reproduce the existing weighted-utility oracle without adding BSRT to it.
    candidate = baseline[baseline.method.isin(["random", "l2", "attention", "learned_token_policy"])]
    oracle = candidate.loc[candidate.groupby(["question_id", "removal_percent"]).utility.idxmax()].copy()
    oracle["method"] = "existing_question_oracle"
    context_means = candidate.groupby(["context_id", "removal_percent", "method"]).f1.mean().reset_index()
    choice = context_means.loc[context_means.groupby(["context_id", "removal_percent"]).f1.idxmax()]
    context_oracle = candidate.merge(choice[["context_id", "removal_percent", "method"]], on=["context_id", "removal_percent", "method"])
    context_oracle["method"] = "context_hindsight_f1"
    combined = pd.concat([combined, original, oracle, context_oracle], ignore_index=True)
    if config.get("comparison_methods"):
        combined = combined[combined.method.isin(config["comparison_methods"])].copy()
    combined.to_csv(destination / "all_comparison_sample_results.csv", index=False)
    summary = summary_tables(combined.to_dict("records"), destination)
    paired = []
    rng = np.random.default_rng(config["seed"])
    for percent in levels:
        for method in ["l2", "learned_token_policy"]:
            subset = combined[combined.removal_percent.eq(percent) & combined.method.isin(["bsrt", method])]
            pivot = subset.pivot(index=["context_id", "question_id"], columns="method", values="f1")
            differences = pivot["bsrt"] - pivot[method]
            groups = [g.to_numpy() for _, g in differences.groupby(level="context_id")]
            sums = np.asarray([g.sum() for g in groups])
            counts = np.asarray([len(g) for g in groups])
            selected = rng.integers(0, len(groups), size=(10000, len(groups)))
            sampled = sums[selected].sum(axis=1)/counts[selected].sum(axis=1)
            low, high = np.quantile(sampled, [.025, .975])
            paired.append({"removal_percent": percent, "comparison": "bsrt minus " + method,
                           "paired_mean_f1_difference": differences.mean(), "ci_low": low, "ci_high": high})
    pd.DataFrame(paired).to_csv(destination / "paired_f1_differences.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    qa.close()
    return summary
