"""Original payload on both sides: uncoded K-copy unicast versus coded multicast."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from bsrt_neural_compression.common import ROOT, digest, read_json, resolve, seed_all, write_json
from bsrt_communication_evaluation.token_ids import decode_ids, received_ids
from semantic_multicast.run import evaluate_case
from .transport import OriginalChannels, allocation


DEFAULT = "multicast_coding_budget/config.json"
FIELDS = ["prediction", "accuracy", "f1", "semantic_similarity"]
KEYS = ["context_id", "group_id", "receiver_count", "channel", "snr_db", "channel_seed", "scenario"]


def prepare(config):
    if config["snr_profile"] != "heterogeneous" or not config["common_contexts_only"]:
        raise ValueError("This controlled experiment uses heterogeneous channels and common contexts across K")
    if set(config["receiver_counts"]) != {2, 3, 5} or set(config["channels"]) != {"awgn", "fading"}:
        raise ValueError("Keep the established K and physical channels")
    if len(set(config["channel_seeds"])) != len(config["channel_seeds"]):
        raise ValueError("Duplicate channel seeds")
    source = resolve(config["cohort_dir"])
    manifests = read_json(source/"groups.json")
    contexts = set.intersection(*(set(g["context_id"] for g in manifests if g["receiver_count"] == k) for k in config["receiver_counts"]))
    contexts = sorted(contexts)
    if config["max_contexts"] is not None:
        contexts = contexts[:int(config["max_contexts"])]
    selected = [g for g in manifests if g["context_id"] in contexts]
    records = pd.read_csv(resolve(config["receiver_csv"]), usecols=["dataset_index", "question", "original_context", "true_answers"])
    records = records.drop_duplicates("dataset_index").set_index("dataset_index")
    from transformers import AutoTokenizer
    reference = read_json(resolve(config["qa_config"]))
    tokenizer = AutoTokenizer.from_pretrained(reference["qa_model"], local_files_only=True)
    groups, payloads = [], {}
    for group in selected:
        rows = [records.loc[int(q)] for q in group["receiver_ids"]]
        text = str(rows[0].original_context)
        if any(row.original_context != text for row in rows):
            raise ValueError("Receiver questions do not share a context")
        if hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16] != group["context_id"]:
            raise ValueError("Original context hash differs from the saved cohort")
        if group["context_id"] not in payloads:
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if not ids or max(ids) >= 2**15:
                raise ValueError("Original IDs do not fit the transport")
            payloads[group["context_id"]] = ids
        questions = [{"id": str(q), "question": str(row.question), "answers": {"text": json.loads(row.true_answers)}}
                     for q, row in zip(group["receiver_ids"], rows, strict=True)]
        groups.append({"id": group["group_id"], "context_id": group["context_id"], "k": group["receiver_count"],
                       "questions": questions, "ids": payloads[group["context_id"]]})
    output = resolve(config["output_dir"])
    if not output.resolve().is_relative_to((ROOT/"multicast_coding_budget/results").resolve()):
        raise ValueError("Use a separate output directory under multicast_coding_budget/results")
    output.mkdir(parents=True, exist_ok=True)
    sources = ["multicast_coding_budget/run.py", "multicast_coding_budget/transport.py", "semantic_multicast/run.py",
               "bsrt_budgeted_ldpc/channel.py", "bsrt_communication_evaluation/channel.py",
               "bsrt_communication_evaluation/token_ids.py", "bsrt_neural_compression/reward.py", "main.py"]
    identity = [{"group_id": g["id"], "context_id": g["context_id"], "K": g["k"],
                 "questions": g["questions"], "payload_ids_sha256": digest(g["ids"]), "original_tokens": len(g["ids"])} for g in groups]
    lock = {"config": config, "groups": identity, "qa_config_sha256": digest(resolve(config["qa_config"]).read_bytes()),
            "implementation": {p: digest((ROOT/p).read_bytes()) for p in sources},
            "software": {p: importlib.metadata.version(p) for p in ["sionna", "torch", "transformers", "numpy"]}}
    lock_path = output/"run_lock.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("Settings or implementation changed: use a different output directory")
    write_json(lock_path, lock)
    factor = len(config["channels"])*len(config["snr_db_values"])*len(config["channel_seeds"])*2
    plan = {"contexts": len(contexts), "groups": len(groups), "distinct_questions": len({q["id"] for g in groups for q in g["questions"]}),
            "expected_cases": factor*len(groups), "expected_receiver_outcomes": factor*sum(g["k"] for g in groups),
            "source_tokens_min": min(map(len, payloads.values())), "source_tokens_max": max(map(len, payloads.values())),
            "removal_percent": 0, "fallback_10_percent_used": False, "training": False,
            "comparison": "uncoded original K-copy unicast vs LDPC original single multicast, aggregate coded length <= unicast budget"}
    allocations = [{"group_id": g["id"], "context_id": g["context_id"], "K": g["k"],
                    **allocation(len(g["ids"]), g["k"], config["max_block_tokens"])} for g in groups]
    write_json(output/"plan.json", plan)
    write_json(output/"budget_allocations.json", allocations)
    return groups, plan, reference


def database(output):
    db = sqlite3.connect(output/"results.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, group_json TEXT, receiver_json TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS metrics (id TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS transports (id TEXT PRIMARY KEY, value TEXT)")
    return db


def export(config, plan, db, counters=None):
    output = resolve(config["output_dir"])
    groups, receivers = [], []
    for key, group_json, receiver_json in db.execute("SELECT id, group_json, receiver_json FROM cases ORDER BY id"):
        group = json.loads(group_json)
        groups.append({"case_id": key, **group})
        common = {field: group[field] for field in KEYS}
        receivers.extend({"case_id": key, **common, **r} for r in json.loads(receiver_json))
    status = {"saved_cases": len(groups), "expected_cases": plan["expected_cases"],
              "saved_receiver_outcomes": len(receivers), "expected_receiver_outcomes": plan["expected_receiver_outcomes"],
              "complete": len(groups) == plan["expected_cases"], "percent_complete": 100*len(groups)/plan["expected_cases"],
              "current_invocation": counters or {}}
    write_json(output/"status.json", status)
    if not groups:
        return
    frame = pd.DataFrame(groups)
    frame.to_csv(output/"group_results.csv", index=False)
    pd.DataFrame(receivers).to_csv(output/"receiver_results.csv", index=False)
    condition = ["receiver_count", "channel", "snr_db", "scenario"]
    metrics = ["mean_accuracy", "mean_f1", "mean_semantic_similarity", "mean_information_ber", "mean_decode_success",
               "all_receivers_decode", "total_qpsk_symbols", "total_transmitted_bits", "budget_bits", "unused_budget_bits", "code_rate"]
    per_k = frame.groupby(condition)[metrics].mean().join(frame.groupby(condition).size().rename("group_seed_outcomes")).reset_index()
    per_k.to_csv(output/"summary_per_k.csv", index=False)
    averaged = per_k.groupby(["channel", "snr_db", "scenario"])[metrics].mean().reset_index()
    averaged.to_csv(output/"summary_averaged_k.csv", index=False)
    from .report import write_report
    write_report(config, plan, status, averaged)


def run(config, dry_run=False):
    groups, plan, reference = prepare(config)
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return
    from bsrt_neural_compression.reward import QAReward
    seed_all(config["selection_seed"], config["threads"])
    output = resolve(config["output_dir"])
    db, qa = database(output), None
    external = []
    counters = {"external_metric_hits": 0, "local_metric_hits": 0, "transport_hits": 0, "new_qa_calls": 0}
    try:
        completed = {row[0] for row in db.execute("SELECT id FROM cases")}
        if len(completed) == plan["expected_cases"]:
            export(config, plan, db, counters)
        else:
            qa = QAReward(reference)
            channels = OriginalChannels(config)
            schema = digest([qa.settings, (ROOT/"main.py").read_bytes().hex(), "all-MiniLM-L6-v2", False])
            legacy_schema = digest([qa.settings, "all-MiniLM-L6-v2", (ROOT/"main.py").read_bytes().hex()])
            for path, table_key, style in [(resolve(config["cohort_dir"])/"results.sqlite3", "id", "current"),
                (resolve(config["original_metrics_dir"])/"transport_and_metrics.sqlite3", "key", "legacy")]:
                if path.exists():
                    external.append((sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True), table_key, style))

            def metric(text, question):
                suffix = [text, question["question"], question["answers"]["text"]]
                key = digest([schema, *suffix])
                saved = db.execute("SELECT value FROM metrics WHERE id=?", (key,)).fetchone()
                if saved:
                    counters["local_metric_hits"] += 1
                    return json.loads(saved[0])
                for source, key_column, style in external:
                    lookup = key if style == "current" else digest([legacy_schema, *suffix])
                    cached = source.execute(f"SELECT value FROM metrics WHERE {key_column}=?", (lookup,)).fetchone()
                    if cached:
                        result = {name: json.loads(cached[0])[name] for name in FIELDS}
                        counters["external_metric_hits"] += 1
                        break
                else:
                    value = qa.metrics(text, [question], allow_no_answer=False, semantic=True)[0]
                    result = {name: value[name] for name in FIELDS}
                db.execute("INSERT INTO metrics VALUES (?,?)", (key, json.dumps(result)))
                return result

            def transport(frame, channel, snr, seed):
                key = digest(["original_segmented_v1", frame.fingerprint, frame.coding, frame.blocks,
                              len(frame.information), channel, snr, seed])
                saved = db.execute("SELECT value FROM transports WHERE id=?", (key,)).fetchone()
                if saved:
                    counters["transport_hits"] += 1
                    value = json.loads(saved[0])
                    return None, value["stats"], value["text"]
                decoded, stats = channels.receive(frame, channel, snr, seed)
                _, usable, invalid = received_ids(decoded, qa.tokenizer)
                stats, text = {**stats, **invalid}, decode_ids(usable, qa.tokenizer)
                db.execute("INSERT INTO transports VALUES (?,?)", (key, json.dumps({"stats": stats, "text": text})))
                return decoded, stats, text

            start = time.monotonic()
            for index, group in enumerate(groups, 1):
                conditions = list(product(config["channels"], config["snr_db_values"], config["channel_seeds"], ["unicast", "multicast"]))
                def case_record(channel, snr, seed, scenario):
                    return {"context_id": group["context_id"], "group_id": group["id"], "receiver_count": group["k"],
                            "channel": channel, "snr_db": float(snr), "channel_seed": seed, "scenario": scenario}
                if all(digest(case_record(*condition)) in completed for condition in conditions):
                    continue
                budget = allocation(len(group["ids"]), group["k"], config["max_block_tokens"])
                # Build one common multicast waveform, never inside the receiver loop.
                original = channels.make_original_frame(group["ids"], group["k"], "uncoded")
                multicast = channels.make_original_frame(group["ids"], group["k"], "ldpc")
                for channel, snr, seed, scenario in conditions:
                    identity = case_record(channel, snr, seed, scenario)
                    key = digest(identity)
                    if key in completed:
                        continue
                    frames = [original]*group["k"] if scenario == "unicast" else [multicast]
                    with db:
                        aggregate, receivers = evaluate_case(config, group, frames, group["questions"], scenario,
                                                            channel, snr, config["snr_profile"], seed, transport, metric)
                        if aggregate["total_transmitted_bits"] > budget["unicast_budget_bits"]:
                            raise AssertionError("Transmitted frame exceeds the K-copy original budget")
                        aggregate.update(identity, source_token_count=len(group["ids"]), removal_percent=0,
                            coding="uncoded" if scenario == "unicast" else "ldpc",
                            budget_bits=budget["unicast_budget_bits"],
                            unused_budget_bits=budget["unicast_budget_bits"]-aggregate["total_transmitted_bits"],
                            code_rate=1. if scenario == "unicast" else budget["effective_multicast_rate"],
                            nominal_multicast_rate=1/group["k"], ldpc_blocks_per_receiver=len(multicast.blocks) if scenario == "multicast" else 0)
                        db.execute("INSERT INTO cases VALUES (?,?,?)", (key, json.dumps(aggregate), json.dumps(receivers)))
                    completed.add(key)
                    counters["new_qa_calls"] = qa.calls
                    if len(completed)%10 == 0:
                        print(f"Saved {len(completed)}/{plan['expected_cases']} cases; group {index}/{len(groups)} "
                              f"K={group['k']} {channel} SNR={snr} {scenario}; QA={qa.calls} "
                              f"external hits={counters['external_metric_hits']}, local hits={counters['local_metric_hits']}, "
                              f"transport hits={counters['transport_hits']}, elapsed={time.monotonic()-start:.0f}s", flush=True)
                        write_json(output/"status.json", {"saved_cases": len(completed), "expected_cases": plan["expected_cases"],
                            "complete": False, "current_invocation": counters, "last_case": identity})
                export(config, plan, db, counters)
            export(config, plan, db, counters)
    finally:
        for source, _, _ in external:
            source.close()
        if qa is not None:
            qa.close()
        db.close()
    if len(completed) == plan["expected_cases"]:
        from .plot import plot_results
        plot_results(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    if args.smoke:
        config.update(output_dir="multicast_coding_budget/results/smoke", max_contexts=1,
                      snr_db_values=[0, 3, 7], channel_seeds=[42])
    if args.plot_only:
        from .plot import plot_results
        plot_results(config)
    else:
        run(config, args.dry_run)


if __name__ == "__main__":
    main()
