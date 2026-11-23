"""Resumable task-aware multicast experiment; never trains or replaces old results."""
from __future__ import annotations

import argparse
from itertools import product
import importlib.metadata
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from bsrt_neural_compression.common import ROOT, digest, read_json, resolve, seed_all, write_json
from bsrt_communication_evaluation.token_ids import decode_ids, received_ids
from paper_multi_receiver_task_alignment.run_task_alignment import angular_co_priority, row_normalize
from .core import BroadcastChannel, allocate_unicast, dimensions, select_ids, stable_seed


DEFAULT = "semantic_multicast/config.json"
METRICS = ["accuracy", "f1", "semantic_similarity", "decode_success", "information_ber", "frame_error"]
CONDITIONS = ["budget_regime", "channel", "coding", "snr_profile", "receiver_count", "removal_percent", "snr_db"]


def validate(config):
    allowed = {"channels": {"awgn", "fading"}, "coding_modes": {"uncoded", "ldpc"},
               "budget_regimes": {"equal_per_representation", "equal_aggregate"},
               "snr_profiles": {"homogeneous", "heterogeneous"}}
    for key, options in allowed.items():
        if not config[key] or not set(config[key]) <= options or len(set(config[key])) != len(config[key]):
            raise ValueError(f"Invalid or duplicate {key}")
    for key in ["receiver_counts", "removal_percentages", "snr_db_values", "channel_seeds"]:
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"Empty or duplicate {key}")
    if any(int(k) != k or k < 2 for k in config["receiver_counts"]):
        raise ValueError("Receiver groups must have at least two members")
    if any(not 0 <= r < 100 for r in config["removal_percentages"]):
        raise ValueError("Invalid removal target")
    if config["heterogeneous_span_db"] < 0 or not all(np.isfinite(config["snr_db_values"])):
        raise ValueError("Invalid SNR configuration")
    dimensions(1, "ldpc", config["ldpc_rate"])
    if config["max_contexts"] is not None and config["max_contexts"] < 1:
        raise ValueError("max_contexts must be positive or null")


def load_groups(config):
    """Select nested receiver subsets without inspecting outcomes or regenerating gradients."""
    source = resolve(config["receiver_csv"])
    records = pd.read_csv(source, usecols=["dataset_index", "question", "original_context", "true_answers"])
    records = records.drop_duplicates("dataset_index").set_index("dataset_index")
    files = sorted(resolve(config["relevance_cache_dir"]).glob("*.npz"))
    if not files:
        raise FileNotFoundError("No cached receiver gradients; no heuristic substitution is permitted")
    groups, manifests = [], []
    used_contexts = 0
    skipped = {str(k): 0 for k in config["receiver_counts"]}
    for path in files:
        if config["max_contexts"] is not None and used_contexts >= config["max_contexts"]:
            break
        with np.load(path, allow_pickle=True) as data:
            if str(data["importance_estimator"].item()) != "distortion_gradient_l2":
                raise ValueError(f"Unexpected importance cache: {path}")
            context_id = str(data["context_id"].item())
            ids = data["receiver_ids"].astype(int)
            tokens = data["token_ids"].astype(int)
            relevance = data["relevance"].astype(float)
            if relevance.shape != (len(ids), len(tokens)) or not np.isfinite(relevance).all() or np.any(relevance < 0):
                raise ValueError(f"Malformed relevance cache: {path}")
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate receivers in context cache")
            context = str(data["original_context"].item())
            for i, receiver in enumerate(ids):
                row = records.loc[int(receiver)]
                if row.original_context != context or row.question != str(data["questions"][i]):
                    raise ValueError("Gradient cache and final evaluation cohort disagree")
                if json.loads(row.true_answers) != json.loads(str(data["true_answers"][i])):
                    raise ValueError("Gradient cache references differ")
            order = np.random.default_rng(stable_seed(config["selection_seed"], context_id)).permutation(len(ids))
            found = False
            for k in config["receiver_counts"]:
                if len(ids) < k:
                    skipped[str(k)] += 1
                    continue
                indices = order[:k]
                if len({str(data["questions"][i]) for i in indices}) != k:
                    raise ValueError("Receiver group contains duplicate questions")
                questions = [{"id": str(int(ids[i])), "question": str(data["questions"][i]),
                              "answers": {"text": json.loads(str(data["true_answers"][i]))}} for i in indices]
                # Every transmitted token ID costs 15 bits. Old UTF-8 density costs do not apply.
                normalized = row_normalize(relevance[indices] / 15.0)
                shared = angular_co_priority(normalized)
                group_id = digest([context_id, [q["id"] for q in questions]])[:20]
                groups.append({"id": group_id, "context_id": context_id, "k": k, "tokens": tokens,
                               "questions": questions, "priorities": normalized, "shared": shared})
                manifests.append({"group_id": group_id, "context_id": context_id, "receiver_count": k,
                    "receiver_ids": [q["id"] for q in questions], "available_receivers": len(ids),
                    "source_tokens": len(tokens), "gradient_cache": str(path.relative_to(ROOT)),
                    "gradient_sha256": digest(path.read_bytes())})
                found = True
            used_contexts += int(found)
    if not groups:
        raise ValueError("No contexts have enough distinct receiver tasks")
    return groups, manifests, skipped


def prepare(config):
    validate(config)
    groups, manifests, skipped = load_groups(config)
    output = resolve(config["output_dir"])
    if not output.resolve().is_relative_to((ROOT/"semantic_multicast/results").resolve()):
        raise ValueError("Use a new directory below semantic_multicast/results to preserve older work")
    output.mkdir(parents=True, exist_ok=True)
    sources = ["semantic_multicast/core.py", "semantic_multicast/run.py",
               "bsrt_budgeted_ldpc/channel.py", "bsrt_communication_evaluation/channel.py",
               "bsrt_communication_evaluation/token_ids.py", "bsrt_neural_compression/reward.py",
               "paper_multi_receiver_task_alignment/run_task_alignment.py", "main.py"]
    lock = {"config": config, "groups": manifests,
            "implementation": {p: digest((ROOT/p).read_bytes()) for p in sources},
            "qa_config_sha256": digest(resolve(config["qa_config"]).read_bytes()),
            "software": {p: importlib.metadata.version(p) for p in ["sionna", "torch", "transformers", "numpy"]}}
    lock_path = output/"run_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        database = output/"results.sqlite3"
        if database.exists():
            with sqlite3.connect(database) as prior:
                if prior.execute("SELECT COUNT(*) FROM cases").fetchone()[0]:
                    raise ValueError("Inputs, code or configuration changed: use a NEW output directory")
    write_json(lock_path, lock)
    factors = math.prod(len(config[k]) for k in ["budget_regimes", "channels", "coding_modes", "snr_profiles",
                                                "removal_percentages", "snr_db_values", "channel_seeds"])
    plan = {"contexts": len({g["context_id"] for g in groups}), "receiver_groups": len(groups),
            "groups_by_K": {str(k): sum(g["k"] == k for g in groups) for k in config["receiver_counts"]},
            "unavailable_groups_by_K": skipped,
            "distinct_questions": len({q["id"] for g in groups for q in g["questions"]}),
            "expected_cases": factors*len(groups)*2,
            "expected_receiver_outcomes": factors*sum(g["k"] for g in groups)*2,
            "training": False, "importance_recomputed": False,
            "coding_comparison": "fixed common rate, no link adaptation",
            "seeds": config["channel_seeds"]}
    write_json(output/"plan.json", plan)
    write_json(output/"groups.json", manifests)
    return groups, plan


def open_results(output):
    db = sqlite3.connect(output/"results.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, group_json TEXT, receiver_json TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS metrics (id TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS transports (id TEXT PRIMARY KEY, value TEXT)")
    return db


def evaluate_case(config, group, frames, questions, scenario, channel, snr, profile, seed, transport, metric):
    offsets = np.zeros(group["k"]) if profile == "homogeneous" else np.linspace(
        -config["heterogeneous_span_db"]/2, config["heterogeneous_span_db"]/2, group["k"])
    receivers = []
    shared = scenario == "multicast"
    if len(frames) != (1 if shared else group["k"]):
        raise AssertionError("Transmission count does not match scenario")
    if shared and len({id(frames[0]) for _ in questions}) != 1:
        raise AssertionError("Multicast must use one shared frame")
    receiver_seeds = [stable_seed(group["context_id"], q["id"], channel, float(snr), profile, seed)
                      for q in questions]
    if len(set(receiver_seeds)) != group["k"]:
        raise AssertionError("Receiver channel seeds collided")
    for index, question in enumerate(questions):
        frame = frames[0] if shared else frames[index]
        actual_snr = float(snr + offsets[index])
        received, stats, text = transport(frame, channel, actual_snr, receiver_seeds[index])
        outcome = metric(text, question)
        receivers.append({"receiver_id": question["id"], "receiver_position": index,
                          "receiver_snr_db": actual_snr, "physical_seed": receiver_seeds[index],
                          "waveform_sha256": frame.fingerprint,
                          "payload_bits": frame.payload_bits, "transmitted_bits": frame.transmitted_bits,
                          "qpsk_symbols": len(frame.symbols), "code_rate": frame.payload_bits/frame.transmitted_bits,
                          "question": question["question"], "references": question["answers"]["text"],
                          "received_context": text, **stats, **outcome})
    aggregate = {"aggregate_information_bits": sum(f.payload_bits for f in frames),
                 "total_transmitted_bits": sum(f.transmitted_bits for f in frames),
                 "total_qpsk_symbols": sum(len(f.symbols) for f in frames),
                 "transmission_count": len(frames),
                 "all_receivers_decode": int(all(r["decode_success"] for r in receivers)),
                 "all_receivers_qa_correct": int(all(r["accuracy"] for r in receivers)),
                 **{f"mean_{m}": float(np.mean([r[m] for r in receivers])) for m in METRICS}}
    if shared and len({r["waveform_sha256"] for r in receivers}) != 1:
        raise AssertionError("Multicast waveform differs by receiver")
    return aggregate, receivers


def export_results(config, plan, db):
    output = resolve(config["output_dir"])
    group_rows, receiver_rows = [], []
    for case, group, receivers in db.execute("SELECT id, group_json, receiver_json FROM cases ORDER BY id"):
        record = json.loads(group)
        group_rows.append({"case_id": case, **record})
        common = {key: record[key] for key in CONDITIONS+["context_id", "group_id", "channel_seed", "scenario"]}
        receiver_rows.extend({"case_id": case, **common, **r} for r in json.loads(receivers))
    count = len(group_rows)
    status = {"saved_cases": count, "expected_cases": plan["expected_cases"],
              "saved_receiver_outcomes": len(receiver_rows), "complete": count == plan["expected_cases"],
              "percent_complete": 100*count/plan["expected_cases"]}
    write_json(output/"status.json", status)
    if not count:
        return
    groups, receivers = pd.DataFrame(group_rows), pd.DataFrame(receiver_rows)
    groups.to_csv(output/"group_results.csv", index=False)
    receivers.to_csv(output/"receiver_results.csv", index=False)
    if groups.case_id.duplicated().any() or receivers.duplicated(["case_id", "receiver_id"]).any():
        raise AssertionError("Duplicate evaluation records")
    pairing = CONDITIONS+["group_id", "channel_seed"]
    left, right = (groups[groups.scenario.eq(s)].set_index(pairing) for s in ["unicast", "multicast"])
    paired = left[["total_qpsk_symbols", "aggregate_information_bits", "mean_accuracy", "mean_f1"]].join(
        right[["total_qpsk_symbols", "aggregate_information_bits", "mean_accuracy", "mean_f1"]],
        lsuffix="_unicast", rsuffix="_multicast", how="inner").reset_index()
    paired["resource_savings"] = 1-paired.total_qpsk_symbols_multicast/paired.total_qpsk_symbols_unicast
    paired["accuracy_difference_multicast_minus_unicast"] = paired.mean_accuracy_multicast-paired.mean_accuracy_unicast
    paired.to_csv(output/"paired_comparison.csv", index=False)
    values = ["aggregate_information_bits", "total_qpsk_symbols", "total_transmitted_bits", "transmission_count",
              "all_receivers_decode", "all_receivers_qa_correct"]+[f"mean_{m}" for m in METRICS]
    summary = groups.groupby(CONDITIONS+["scenario"])[values].mean().join(
        groups.groupby(CONDITIONS+["scenario"]).size().rename("group_seed_outcomes")).reset_index()
    summary.to_csv(output/"summary.csv", index=False)
    # Do not conflate macro group means with task-weighted pooled receiver means.
    receivers.groupby(CONDITIONS+["scenario"])[METRICS+["ldpc_block_error", "raw_coded_ber"]].mean().reset_index().to_csv(
        output/"receiver_summary.csv", index=False)
    paired.groupby(CONDITIONS)[["resource_savings", "accuracy_difference_multicast_minus_unicast"]].mean().reset_index().to_csv(
        output/"paired_summary.csv", index=False)
    from .report import write_report
    write_report(config, plan, status, summary, paired)


def run(config, dry_run=False):
    groups, plan = prepare(config)
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return
    from bsrt_neural_compression.reward import QAReward
    seed_all(config["selection_seed"], config["threads"])
    output = resolve(config["output_dir"])
    reference = read_json(resolve(config["qa_config"]))
    qa, db = None, open_results(output)
    try:
        completed = {row[0] for row in db.execute("SELECT id FROM cases")}
        if len(completed) == plan["expected_cases"]:
            export_results(config, plan, db)
            print("All cases already saved; no channel/QA evaluation repeated.", flush=True)
            return
        qa = QAReward(reference)
        for group in groups:
            if any(t in qa.tokenizer.all_special_ids for t in group["tokens"]):
                raise ValueError("Cached source unexpectedly contains special tokens")
            if any(t >= len(qa.tokenizer) or t < 0 for t in group["tokens"]):
                raise ValueError("Cached token vocabulary differs from the QA tokenizer")
        channels = BroadcastChannel(config)
        metric_schema = digest([qa.settings, (ROOT/"main.py").read_bytes().hex(), "all-MiniLM-L6-v2", False])
        qa_hits = 0

        def metric(text, question):
            nonlocal qa_hits
            key = digest([metric_schema, text, question["question"], question["answers"]["text"]])
            cached = db.execute("SELECT value FROM metrics WHERE id=?", (key,)).fetchone()
            if cached:
                qa_hits += 1
                return json.loads(cached[0])
            result = qa.metrics(text, [question], allow_no_answer=False, semantic=True)[0]
            result = {k: result[k] for k in ["prediction", "accuracy", "f1", "semantic_similarity"]}
            db.execute("INSERT INTO metrics VALUES (?,?)", (key, json.dumps(result)))
            return result

        def transport(frame, channel, snr, seed):
            key = digest([frame.fingerprint, frame.coding, len(frame.information), channel, snr, seed])
            cached = db.execute("SELECT value FROM transports WHERE id=?", (key,)).fetchone()
            if cached:
                value = json.loads(cached[0])
                return None, value["stats"], value["text"]
            decoded, stats = channels.receive(frame, channel, snr, seed)
            _, usable, invalid = received_ids(decoded, qa.tokenizer)
            stats, text = {**stats, **invalid}, decode_ids(usable, qa.tokenizer)
            db.execute("INSERT INTO transports VALUES (?,?)", (key, json.dumps({"stats": stats, "text": text})))
            return decoded, stats, text

        started = time.monotonic()
        for number, group in enumerate(groups, 1):
            for regime, removal, coding in product(config["budget_regimes"], config["removal_percentages"], config["coding_modes"]):
                keep = max(1, len(group["tokens"])-int(len(group["tokens"])*removal/100))
                counts = allocate_unicast(keep, group["k"], coding, config["ldpc_rate"], regime)
                # One shared encode/modulate operation outside ALL receiver/channel loops.
                multicast = [channels.make_frame(select_ids(group["tokens"], group["shared"], keep), coding, config["ldpc_rate"])]
                unicast = [channels.make_frame(select_ids(group["tokens"], scores, n), coding, config["ldpc_rate"])
                           for scores, n in zip(group["priorities"], counts, strict=True)]
                unicast_bits = sum(f.transmitted_bits for f in unicast)
                multicast_bits = multicast[0].transmitted_bits
                if regime == "equal_aggregate" and unicast_bits > multicast_bits:
                    raise AssertionError("Unicast exceeds the shared aggregate resource constraint")
                for channel, profile, snr, seed, scenario in product(config["channels"], config["snr_profiles"],
                        config["snr_db_values"], config["channel_seeds"], ["unicast", "multicast"]):
                    key_record = {"group_id": group["id"], "context_id": group["context_id"],
                        "receiver_count": group["k"], "budget_regime": regime, "removal_percent": removal,
                        "coding": coding, "channel": channel, "snr_profile": profile,
                        "snr_db": float(snr), "channel_seed": seed, "scenario": scenario}
                    key = digest(key_record)
                    if key in completed:
                        continue
                    frames = multicast if scenario == "multicast" else unicast
                    with db:
                        aggregate, outcomes = evaluate_case(config, group, frames, group["questions"], scenario,
                                                           channel, snr, profile, seed, transport, metric)
                        aggregate.update(key_record, aggregate_ceiling_bits=multicast_bits,
                            unused_aggregate_bits=multicast_bits-unicast_bits if regime == "equal_aggregate" and scenario == "unicast" else 0,
                            multicast_kept_tokens=keep, unicast_kept_tokens=counts,
                            source_token_count=len(group["tokens"]))
                        db.execute("INSERT INTO cases VALUES (?,?,?)", (key, json.dumps(aggregate), json.dumps(outcomes)))
                    completed.add(key)
                    if len(completed) % 10 == 0:
                        print(f"Saved {len(completed)}/{plan['expected_cases']} cases; group {number}/{len(groups)} "
                              f"K={group['k']} {channel} {coding} {profile} SNR={snr}; "
                              f"new QA={qa.calls}, metric hits={qa_hits}, elapsed={time.monotonic()-started:.0f}s", flush=True)
                        write_json(output/"status.json", {"saved_cases": len(completed),
                            "expected_cases": plan["expected_cases"], "complete": False,
                            "last_case": key_record, "new_qa_calls": qa.calls, "metric_hits": qa_hits})
            export_results(config, plan, db)
        export_results(config, plan, db)
    finally:
        if qa is not None:
            qa.close()
        db.close()
    from .plot import plot_results
    plot_results(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--smoke", action="store_true", help="One context, K=2, two SNRs, both channels/coding modes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    if args.smoke:
        config.update(output_dir="semantic_multicast/results/smoke", max_contexts=1, receiver_counts=[2],
                      removal_percentages=[30], snr_db_values=[3, 15], channel_seeds=[42])
    if args.plot_only:
        from .plot import plot_results
        plot_results(config)
    else:
        run(config, args.dry_run)


if __name__ == "__main__":
    main()
