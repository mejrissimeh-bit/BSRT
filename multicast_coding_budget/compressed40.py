"""40% semantic-aligned multicast versus cached original uncoded unicast."""
import argparse
import importlib.metadata
from itertools import product
import json
import sqlite3
import time

import numpy as np
import pandas as pd

from bsrt_neural_compression.common import ROOT, digest, read_json, resolve, seed_all, write_json
from bsrt_communication_evaluation.token_ids import decode_ids, received_ids
from paper_multi_receiver_task_alignment.run_task_alignment import (
    angular_co_priority, row_normalize, task_distortion_gradient_l2)
from semantic_multicast.core import select_ids
from semantic_multicast.run import evaluate_case
from .compressed_transport import CompressedChannels, compressed_allocation
from .run import database, FIELDS, KEYS


BASE = "multicast_coding_budget/results/full"
DEFAULT_OUTPUT = "multicast_coding_budget/results/compressed40"


def prepare(output_dir, smoke=False):
    base = resolve(BASE)
    original_lock = read_json(base/"run_lock.json")
    if not read_json(base/"status.json")["complete"]:
        raise ValueError("The matched original evaluation must be complete")
    for path, expected in original_lock["implementation"].items():
        if digest((ROOT/path).read_bytes()) != expected:
            raise ValueError(f"Original simulation implementation changed: {path}")
    if any(importlib.metadata.version(p) != v for p, v in original_lock["software"].items()):
        raise ValueError("Original simulation software versions changed")
    config = dict(original_lock["config"])
    config.update(output_dir=output_dir, removal_percent=40)
    if smoke:
        config.update(snr_db_values=[0, 3], channel_seeds=[42])
    output = resolve(output_dir)
    if not output.resolve().is_relative_to((ROOT/"multicast_coding_budget/results").resolve()) or output == base:
        raise ValueError("Use a separate output directory below multicast_coding_budget/results")
    output.mkdir(parents=True, exist_ok=True)
    reference = read_json(resolve(config["qa_config"]))
    if digest(resolve(config["qa_config"]).read_bytes()) != original_lock["qa_config_sha256"]:
        raise ValueError("QA configuration changed")
    records = pd.read_csv(resolve(config["receiver_csv"]),
                          usecols=["dataset_index", "question", "original_context", "true_answers"])
    records = records.drop_duplicates("dataset_index").set_index("dataset_index")
    manifests = {g["group_id"]: g for g in read_json(resolve(config["cohort_dir"])/"groups.json")}
    identities = original_lock["groups"]
    if smoke:
        first = identities[0]["context_id"]
        identities = [g for g in identities if g["context_id"] == first]
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained(reference["qa_model"], local_files_only=True)
    model = None
    gradients, cache_sources, groups, allocations = {}, {}, [], []
    for identity in identities:
        context_id, group_id, k = identity["context_id"], identity["group_id"], identity["K"]
        questions = identity["questions"]
        for q in questions:
            row = records.loc[int(q["id"])]
            if q["question"] != row.question or q["answers"]["text"] != json.loads(row.true_answers):
                raise ValueError("Archived receiver question/reference changed")
        context = str(records.loc[int(questions[0]["id"])].original_context)
        ids = tokenizer.encode(context, add_special_tokens=False, truncation=False)
        if digest(ids) != identity["payload_ids_sha256"]:
            raise ValueError("Original token IDs changed")
        manifest = manifests[group_id]
        if manifest["receiver_ids"] != [q["id"] for q in questions]:
            raise ValueError("Receiver group changed")
        source = resolve(manifest["gradient_cache"])
        if digest(source.read_bytes()) != manifest["gradient_sha256"]:
            raise ValueError("Saved alignment cache changed")
        with np.load(source, allow_pickle=True) as data:
            if str(data["importance_estimator"].item()) != "distortion_gradient_l2":
                raise ValueError("Unexpected alignment importance")
            if context != str(data["original_context"].item()):
                raise ValueError("Gradient context mismatch")
            if np.array_equal(data["token_ids"], ids):
                lookup = {str(int(q)): i for i, q in enumerate(data["receiver_ids"])}
                relevance = np.asarray([data["relevance"][lookup[q["id"]]] for q in questions], dtype=float)
                cache_sources[context_id] = {"path": str(source.relative_to(ROOT)),
                                             "sha256": digest(source.read_bytes()), "full_length_recomputed": False}
            else:
                # Only two old caches truncate the full original context at 256 tokens.
                if context_id not in gradients:
                    folder = output/"full_length_relevance"
                    folder.mkdir(exist_ok=True)
                    path = folder/(context_id+".npz")
                    largest = max((g for g in identities if g["context_id"] == context_id), key=lambda g: g["K"])
                    qs = largest["questions"]
                    signature = digest([digest(source.read_bytes()), identity["payload_ids_sha256"], qs,
                                        reference, digest((ROOT/"paper_multi_receiver_task_alignment/run_task_alignment.py").read_bytes())])
                    if path.exists():
                        with np.load(path, allow_pickle=False) as full:
                            if str(full["signature"].item()) != signature:
                                raise ValueError("Full-length gradient cache provenance changed")
                            gradients[context_id] = dict(zip(full["receiver_ids"].tolist(), full["relevance"], strict=True))
                    else:
                        if model is None:
                            model = AutoModelForQuestionAnswering.from_pretrained(reference["qa_model"], local_files_only=True)
                            model.eval()
                        canonical = tokenizer.decode(ids, skip_special_tokens=True)
                        values = []
                        for q in qs:
                            paired = tokenizer(q["question"], canonical, truncation="only_second", max_length=512)
                            if sum(s == 1 for s in paired.sequence_ids()) != len(ids):
                                raise ValueError("Full context does not fit the gradient model; do not silently truncate")
                            values.append(task_distortion_gradient_l2(q["question"], q["answers"]["text"],
                                                                      canonical, ids, tokenizer, model))
                        np.savez_compressed(path, signature=np.asarray(signature),
                                            receiver_ids=np.asarray([q["id"] for q in qs]), relevance=np.asarray(values))
                        gradients[context_id] = dict(zip([q["id"] for q in qs], values, strict=True))
                        print(f"Extended cached relevance to full context {context_id}: {len(ids)} tokens", flush=True)
                    cache_sources[context_id] = {"path": str(path.relative_to(ROOT)),
                                                 "sha256": digest(path.read_bytes()), "full_length_recomputed": True}
                relevance = np.asarray([gradients[context_id][q["id"]] for q in questions])
        shared = angular_co_priority(row_normalize(relevance/15.))
        kept = max(1, len(ids)-int(len(ids)*.4))
        selected = select_ids(ids, shared, kept)
        group = {"id": group_id, "context_id": context_id, "k": k, "questions": questions,
                 "ids": selected, "original_tokens": len(ids)}
        groups.append(group)
        allocations.append({"group_id": group_id, "context_id": context_id, "K": k,
                            "selected_ids": selected, **compressed_allocation(len(ids), kept, k)})
    del model
    sources = ["multicast_coding_budget/compressed40.py", "multicast_coding_budget/compressed_transport.py",
               "paper_multi_receiver_task_alignment/run_task_alignment.py", "semantic_multicast/core.py"]
    lock = {"config": config, "original_run_lock_sha256": digest((base/"run_lock.json").read_bytes()),
            "original_database_sha256": digest((base/"results.sqlite3").read_bytes()),
            "groups": allocations, "relevance_caches": cache_sources,
            "implementation": {p: digest((ROOT/p).read_bytes()) for p in sources}}
    path = output/"run_lock.json"
    if path.exists() and read_json(path) != lock:
        result_path = output/"results.sqlite3"
        if result_path.exists():
            with sqlite3.connect(result_path) as saved:
                if saved.execute("SELECT COUNT(*) FROM cases").fetchone()[0]:
                    raise ValueError("Experiment changed; use a different output directory")
    write_json(path, lock)
    factor = len(config["channels"])*len(config["snr_db_values"])*len(config["channel_seeds"])*2
    plan = {"contexts": len({g["context_id"] for g in groups}), "groups": len(groups),
            "distinct_questions": len({q["id"] for g in groups for q in g["questions"]}),
            "expected_cases": factor*len(groups), "expected_receiver_outcomes": factor*sum(g["k"] for g in groups),
            "multicast_removal_percent": 40, "unicast_removal_percent": 0,
            "importance": "cached answer-supervised task gradients and angular semantic alignment; not question-blind",
            "reused_full_length_context_caches": sum(not v["full_length_recomputed"] for v in cache_sources.values()),
            "extended_full_length_context_caches": sum(v["full_length_recomputed"] for v in cache_sources.values())}
    write_json(output/"plan.json", plan)
    write_json(output/"budget_allocations.json", allocations)
    return config, groups, plan, reference


def export(config, plan, db, counters):
    output = resolve(config["output_dir"])
    groups, receivers = [], []
    for key, gj, rj in db.execute("SELECT id, group_json, receiver_json FROM cases ORDER BY id"):
        g = json.loads(gj)
        groups.append({"case_id": key, **g})
        receivers.extend({"case_id": key, **{k: g[k] for k in KEYS},
                          "removal_percent": g["removal_percent"], **r} for r in json.loads(rj))
    status = {"saved_cases": len(groups), "expected_cases": plan["expected_cases"],
              "saved_receiver_outcomes": len(receivers), "complete": len(groups) == plan["expected_cases"],
              "current_invocation": counters}
    write_json(output/"status.json", status)
    if groups:
        frame = pd.DataFrame(groups)
        frame.to_csv(output/"group_results.csv", index=False)
        pd.DataFrame(receivers).to_csv(output/"receiver_results.csv", index=False)
        metrics = ["mean_accuracy", "mean_f1", "mean_semantic_similarity", "mean_information_ber",
                   "mean_decode_success", "total_transmitted_bits", "budget_bits", "code_rate"]
        per_k = frame.groupby(["receiver_count", "channel", "snr_db", "scenario"])[metrics].mean().reset_index()
        per_k.to_csv(output/"summary_per_k.csv", index=False)
        per_k.groupby(["channel", "snr_db", "scenario"])[metrics].mean().reset_index().to_csv(
            output/"summary_averaged_k.csv", index=False)


def run(output_dir=DEFAULT_OUTPUT, smoke=False, dry_run=False):
    seed_all(20261105, 2)
    config, groups, plan, reference = prepare(output_dir, smoke)
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return
    output = resolve(output_dir)
    db = database(output)
    base = sqlite3.connect((resolve(BASE)/"results.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)
    old = sqlite3.connect((resolve(config["cohort_dir"])/"results.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)
    counters = {"imported_original_cases": 0, "external_transport_hits": 0,
                "local_transport_hits": 0, "external_metric_hits": 0, "new_qa_calls": 0}
    qa = None
    try:
        completed = {r[0] for r in db.execute("SELECT id FROM cases")}
        conditions = list(product(config["channels"], config["snr_db_values"], config["channel_seeds"]))
        def identity(g, channel, snr, seed, scenario):
            return {"context_id": g["context_id"], "group_id": g["id"], "receiver_count": g["k"],
                    "channel": channel, "snr_db": float(snr), "channel_seed": seed, "scenario": scenario}
        # Baseline is imported exactly, not reevaluated or replaced by SNR-matched averages.
        with db:
            for g in groups:
                for c, snr, seed in conditions:
                    key = digest(identity(g, c, snr, seed, "unicast"))
                    if key not in completed:
                        row = base.execute("SELECT group_json, receiver_json FROM cases WHERE id=?", (key,)).fetchone()
                        if not row:
                            raise ValueError("Missing exact matched original-unicast case")
                        db.execute("INSERT INTO cases VALUES (?,?,?)", (key, *row))
                        completed.add(key)
                        counters["imported_original_cases"] += 1
        export(config, plan, db, counters)
        if len(completed) == plan["expected_cases"]:
            return
        from bsrt_neural_compression.reward import QAReward
        qa = QAReward(reference)
        channels = CompressedChannels(config)
        schema = digest([qa.settings, (ROOT/"main.py").read_bytes().hex(), "all-MiniLM-L6-v2", False])

        def metric(text, q):
            key = digest([schema, text, q["question"], q["answers"]["text"]])
            saved = db.execute("SELECT value FROM metrics WHERE id=?", (key,)).fetchone()
            if saved:
                return json.loads(saved[0])
            for source in (base, old):
                saved = source.execute("SELECT value FROM metrics WHERE id=?", (key,)).fetchone()
                if saved:
                    result = {f: json.loads(saved[0])[f] for f in FIELDS}
                    counters["external_metric_hits"] += 1
                    break
            else:
                result = {f: v for f, v in qa.metrics(text, [q], allow_no_answer=False, semantic=True)[0].items() if f in FIELDS}
            db.execute("INSERT INTO metrics VALUES (?,?)", (key, json.dumps(result)))
            return result

        def transport(frame, channel, snr, seed):
            key = digest(["compressed40_budget_v1", frame.fingerprint, frame.blocks, channel, snr, seed])
            saved = db.execute("SELECT value FROM transports WHERE id=?", (key,)).fetchone()
            if saved:
                counters["local_transport_hits"] += 1
                v = json.loads(saved[0])
                return None, v["stats"], v["text"]
            # Older transport is reusable only for identical payload, dimensions and waveform.
            if len(frame.blocks) == 1 and frame.blocks[0][1] == frame.blocks[0][2]:
                k, n, _ = frame.blocks[0]
                candidates = [(base, digest(["original_segmented_v1", frame.fingerprint, "ldpc", ((k, n),), k, channel, snr, seed])),
                              (old, digest([frame.fingerprint, "ldpc", k, channel, snr, seed]))]
                for source, lookup in candidates:
                    saved = source.execute("SELECT value FROM transports WHERE id=?", (lookup,)).fetchone()
                    if saved:
                        value = json.loads(saved[0])
                        stats = value["stats"]
                        stats.update(ldpc_blocks=1, ldpc_failed_blocks=int(stats["frame_error"]))
                        db.execute("INSERT INTO transports VALUES (?,?)", (key, json.dumps(value)))
                        counters["external_transport_hits"] += 1
                        return None, stats, value["text"]
            decoded, stats = channels.receive(frame, channel, snr, seed)
            _, usable, invalid = received_ids(decoded, qa.tokenizer)
            stats, text = {**stats, **invalid}, decode_ids(usable, qa.tokenizer)
            db.execute("INSERT INTO transports VALUES (?,?)", (key, json.dumps({"stats": stats, "text": text})))
            return decoded, stats, text

        started = time.monotonic()
        for index, g in enumerate(groups, 1):
            pending = [(c, snr, seed) for c, snr, seed in conditions
                       if digest(identity(g, c, snr, seed, "multicast")) not in completed]
            if not pending:
                continue
            frame = channels.make_compressed_frame(g["ids"], g["original_tokens"], g["k"])
            budget = compressed_allocation(g["original_tokens"], len(g["ids"]), g["k"])
            for c, snr, seed in pending:
                record = identity(g, c, snr, seed, "multicast")
                key = digest(record)
                with db:
                    aggregate, receivers = evaluate_case(config, g, [frame], g["questions"], "multicast",
                                                        c, snr, config["snr_profile"], seed, transport, metric)
                    if aggregate["total_transmitted_bits"] != budget["budget_bits"]:
                        raise AssertionError("Compressed multicast must use exactly the original-unicast budget")
                    aggregate.update(record, source_token_count=g["original_tokens"], kept_token_count=len(g["ids"]),
                        removal_percent=40, actual_removal_percent=budget["actual_removal_percent"],
                        coding="ldpc_plus_repetition" if budget["repeated_bits"] else "ldpc",
                        repeated_bits=budget["repeated_bits"], budget_bits=budget["budget_bits"],
                        unused_budget_bits=0, code_rate=budget["effective_code_rate"])
                    db.execute("INSERT INTO cases VALUES (?,?,?)", (key, json.dumps(aggregate), json.dumps(receivers)))
                completed.add(key)
                counters["new_qa_calls"] = qa.calls
                if len(completed)%10 == 0:
                    print(f"Saved {len(completed)}/{plan['expected_cases']}; group {index}/{len(groups)} K={g['k']} "
                          f"{c} SNR={snr}; QA={qa.calls}; elapsed={time.monotonic()-started:.0f}s", flush=True)
                    write_json(output/"status.json", {"saved_cases": len(completed), "expected_cases": plan["expected_cases"],
                                                      "complete": False, "current_invocation": counters})
            export(config, plan, db, counters)
        export(config, plan, db, counters)
    finally:
        if qa is not None:
            qa.close()
        db.close()
        base.close()
        old.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.output, args.smoke, args.dry_run)
    if not args.dry_run:
        from .audit_compressed40 import audit
        from .plot_compressed40 import plot
        audit(args.output)
        plot(args.output)
