from __future__ import annotations

import copy
import csv
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import digest, read_json, resolve, seed_all, write_json
from .data import split_manifest, training_questions
from .model import BSRT, ContextFeatures, RetentionNetwork, render_context
from .reward import QAReward
from .search import optimize_mask


def span_targets(record, feature):
    targets = np.zeros(len(feature["input_ids"]), dtype=np.float32)
    for question in record["questions"]:
        for start, text in zip(question["answers"]["answer_start"], question["answers"]["text"], strict=True):
            end = start + len(text)
            for i, (left, right) in enumerate(feature["offsets"]):
                if left < end and right > start:
                    targets[i] = 1
    return targets


def checkpoint(path, network, optimizer, config, epoch, stage, score=None):
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": network.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "epoch": epoch, "stage": stage, "validation_f1": score,
                "torch_rng": torch.get_rng_state()}, temporary)
    temporary.replace(path)


def validate(policy, records, qa, percentages):
    rows = []
    for record in records:
        for percent in percentages:
            # Compression is complete before any questions are passed to QA.
            text = policy.compress(record["context"], percent/100)
            metrics = qa.reward(text, record["questions"])
            rows.append({"context_id": record["context_id"], "percent": percent, **metrics})
    return float(np.mean([row["f1"] for row in rows])), rows


def log_jsonl(path, data):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data) + "\n")


def fit_stage(network, features, qa, config, records, validation, stage, targets=None):
    output = resolve(config["output_dir"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    policy = BSRT(network, features)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"], weight_decay=.01)
    last = checkpoints / f"{stage}_last.pt"
    best_path = checkpoints / "best.pt"
    stage_best_path = checkpoints / f"{stage}_best.pt"
    start_epoch = 0
    best_score = -1.0
    stage_best_score = -1.0
    if stage_best_path.exists():
        stage_best_score = torch.load(stage_best_path, weights_only=False, map_location="cpu")["validation_f1"]
    if best_path.exists():
        best_score = torch.load(best_path, weights_only=False, map_location="cpu")["validation_f1"]
    if last.exists():
        saved = torch.load(last, weights_only=False, map_location="cpu")
        network.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        start_epoch = saved["epoch"]
    epochs = config["pretrain_epochs"] if stage == "pretrain" else config["distill_epochs"]
    examples = [(record, percent) for record in records for percent in config["removal_percentages"]]
    device = next(network.parameters()).device
    for epoch in range(start_epoch, epochs):
        rng = np.random.default_rng(config["seed"] + epoch + int(digest(stage)[:6], 16))
        order = rng.permutation(len(examples))
        network.train()
        losses = []
        start = time.perf_counter()
        for offset in order:
            record, percent = examples[offset]
            feature = features.get(record["context"])
            optimizer.zero_grad(set_to_none=True)
            logits = network(feature["embeddings"].to(device).float(), feature["structure"].to(device), percent/100)
            if config.get("answer_span_supervision", True):
                answer = torch.tensor(span_targets(record, feature), device=device)
                positive_weight = min(6.0, max(1.0, float((1-answer.mean()) / answer.mean().clamp(min=.01))))
                auxiliary = F.binary_cross_entropy_with_logits(logits, answer, pos_weight=logits.new_tensor(positive_weight))
            else:
                auxiliary = logits.sum()*0
            if targets is None or (record["context_id"], percent) not in targets:
                loss = auxiliary
            else:
                samples = targets[(record["context_id"], percent)]
                target = np.average([sample["target"] for sample in samples], axis=0,
                                    weights=[sample["weight"] for sample in samples])
                target = torch.tensor(target, dtype=torch.float32, device=device)
                loss = F.binary_cross_entropy_with_logits(logits, target) + config["auxiliary_weight"] * auxiliary
            budget_loss = (logits.sigmoid().mean() - (1-percent/100))**2
            loss = loss + config["budget_weight"]*budget_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        network.eval()
        score, rows = validate(policy, validation, qa, config["removal_percentages"])
        event = {"stage": stage, "epoch": epoch+1, "loss": float(np.mean(losses)),
                 "validation_context_mean_f1": score, "seconds": time.perf_counter()-start,
                 "new_qa_calls": qa.calls}
        print(json.dumps(event), flush=True)
        log_jsonl(output / "training_history.jsonl", event)
        write_json(output / f"validation_{stage}_epoch_{epoch+1}.json", rows)
        checkpoint(last, network, optimizer, config, epoch+1, stage, score)
        if score > stage_best_score:
            stage_best_score = score
            checkpoint(stage_best_path, network, optimizer, config, epoch+1, stage, score)
        if score > best_score:
            best_score = score
            checkpoint(best_path, network, optimizer, config, epoch+1, stage, score)
    return policy


def train(config, qa=None):
    import psutil
    manifest = split_manifest(config)
    output = resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "progress.json"
    state = read_json(state_path) if state_path.exists() else {"pretraining_done": False, "rounds_done": 0}
    write_json(output / "run_config.json", config)
    seed_all(config["seed"], config["threads"])
    started = time.perf_counter()
    owns_qa = qa is None
    qa = qa or QAReward(config)
    feature = ContextFeatures(config, qa.model, qa.tokenizer)
    network = RetentionNetwork(**config["model"]).to(qa.device)
    if config.get("match_reference_initialization"):
        reference_model = dict(config["model"])
        reference_model.update(contextual=True, budget_conditioning=True, structural_features=True)
        with torch.random.fork_rng():
            torch.manual_seed(config["seed"])
            initial = RetentionNetwork(**reference_model).state_dict()
        state_dict = network.state_dict()
        for key, value in initial.items():
            if not key.startswith("transformer.") and key in state_dict and state_dict[key].shape == value.shape:
                state_dict[key] = value.to(qa.device)
        network.load_state_dict(state_dict)
    best_path = output / "checkpoints" / "best.pt"
    if best_path.exists():
        network.load_state_dict(torch.load(best_path, weights_only=False, map_location=qa.device)["state_dict"])
    train_rows, validation = manifest["train"], manifest["validation"]
    for i, row in enumerate(train_rows + validation, 1):
        feature.get(row["context"])
        if i % 8 == 0:
            print(f"Cached context features {i}/{len(train_rows)+len(validation)}", flush=True)
    if not best_path.exists() and config.get("initialize_from"):
        source = resolve(config["initialize_from"])
        initial = torch.load(source, map_location=qa.device, weights_only=False)
        if initial["config"]["model"] != config["model"]:
            raise ValueError("Warm-start architecture differs from the configured retention network")
        network.load_state_dict(initial["state_dict"])
        score, rows = validate(BSRT(network, feature), validation, qa, config["removal_percentages"])
        optimizer = torch.optim.AdamW(network.parameters(), lr=config["learning_rate"])
        best_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint(best_path, network, optimizer, config, 0, "warm_start", score)
        write_json(output / "validation_warm_start.json", rows)
        write_json(output / "initialization.json", {"source": str(source), "source_sha256": digest(source.read_bytes()), "validation_f1": score})
        print(f"Warm-start validation mean F1 across all budgets: {score:.4f}", flush=True)
    if not state["pretraining_done"]:
        if config["pretrain_epochs"]:
            fit_stage(network, feature, qa, config, train_rows, validation, "pretrain")
        else:
            # Ablation checkpoint must still be selected before any test use.
            optimizer = torch.optim.AdamW(network.parameters())
            score = -1.0
            if config.get("initial_checkpoint_eligible", True):
                score, _ = validate(BSRT(network, feature), validation, qa, config["removal_percentages"])
            best_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint(best_path, network, optimizer, config, 0, "no_pretraining", score)
        state["pretraining_done"] = True
        write_json(state_path, state)
    for round_index in range(state["rounds_done"], config["improvement_rounds"]):
        network.load_state_dict(torch.load(best_path, map_location=qa.device, weights_only=False)["state_dict"])
        policy = BSRT(network, feature)
        targets = {}
        # Earlier round targets remain available to prevent forgetting.
        for past in range(round_index+1):
            for i, row in enumerate(train_rows, 1):
                for percent in config.get("search_percentages", config["removal_percentages"]):
                    path = output / "search" / f"round_{past}" / f"{row['context_id']}_{percent}.json"
                    if not path.exists():
                        print(f"BSRT search round {past+1} context {i}/{len(train_rows)} removal={percent}%", flush=True)
                    value = optimize_mask(policy, row, training_questions(row, config), percent, qa, config, path, past)
                    targets.setdefault((row["context_id"], percent), []).append(value)
        fit_stage(network, feature, qa, config, train_rows, validation, f"distill_{round_index}", targets)
        state["rounds_done"] = round_index+1
        write_json(state_path, state)
    saved = torch.load(best_path, weights_only=False, map_location="cpu")
    summary = {"best_validation_f1": saved["validation_f1"], "best_stage": saved["stage"],
               "best_epoch": saved["epoch"], "elapsed_seconds_this_invocation": time.perf_counter()-started,
               "new_qa_calls_this_invocation": qa.calls, "qa_seconds": qa.seconds,
               "peak_rss_mb": psutil.Process().memory_info().peak_wset / 1e6 if hasattr(psutil.Process().memory_info(), "peak_wset") else psutil.Process().memory_info().rss/1e6,
               "gpu_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
               "train_contexts": len(train_rows), "validation_contexts": len(validation),
               "train_questions_available": sum(len(r["questions"]) for r in train_rows),
               "train_questions_for_reward": sum(len(training_questions(r, config)) for r in train_rows)}
    write_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if owns_qa:
        qa.close()
    return best_path
