"""Run/resume clustered multicast vs individual unicast on all 500 questions."""
import argparse
from itertools import product
import json
import sqlite3
import time

import numpy as np
import pandas as pd

from bsrt_neural_compression.common import ROOT, digest, read_json, resolve, seed_all, write_json
from bsrt_communication_evaluation.token_ids import decode_ids, received_ids
from semantic_multicast.core import stable_seed
from .cohort import prepare
from .transport import Channels, budget_for_context, cluster_allocation


FIELDS = ["prediction", "accuracy", "f1", "semantic_similarity"]
METRICS = ["accuracy", "f1", "semantic_similarity", "information_ber", "decode_success"]


def export(config, plan, db, counters):
    output = resolve(config["output_dir"])
    groups, receivers = [], []
    for key, gj, rj in db.execute("SELECT id, group_json, receiver_json FROM cases ORDER BY id"):
        g = json.loads(gj)
        groups.append({"case_id": key, **g})
        receivers.extend({"case_id": key, **{f: g[f] for f in ["context_id", "channel", "snr_db", "channel_seed", "scenario"]},
                          **r} for r in json.loads(rj))
    status = {"saved_cases": len(groups), "expected_cases": plan["expected_cases"],
              "saved_receiver_outcomes": len(receivers), "expected_receiver_outcomes": plan["expected_receiver_outcomes"],
              "complete": len(groups) == plan["expected_cases"], "current_invocation": counters}
    write_json(output/"status.json", status)
    if groups:
        pd.DataFrame(groups).to_csv(output/"context_results.csv", index=False)
        frame = pd.DataFrame(receivers)
        frame.to_csv(output/"receiver_results.csv", index=False)
        # Each question has equal weight; there is no equal-context or equal-K averaging.
        per_seed = frame.groupby(["channel", "snr_db", "channel_seed", "scenario"])[METRICS].mean().reset_index()
        per_seed.to_csv(output/"summary_per_seed.csv", index=False)
        per_seed.groupby(["channel", "snr_db", "scenario"])[METRICS].mean().reset_index().to_csv(
            output/"summary_all_questions.csv", index=False)
    clean = [json.loads(r[0]) for r in db.execute("SELECT value FROM clean")]
    if clean:
        frame = pd.DataFrame(clean)
        frame.to_csv(output/"clean_receiver_results.csv", index=False)
        frame.groupby("scenario")[["accuracy", "f1", "semantic_similarity"]].mean().reset_index().to_csv(
            output/"clean_summary.csv", index=False)


def run(config, smoke=False, dry_run=False):
    seed_all(config["selection_seed"], config["threads"])
    groups, plan, reference = prepare(config, smoke)
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return
    output = resolve(config["output_dir"])
    db = sqlite3.connect(output/"results.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    for table in ["metrics", "transports", "clean"]:
        db.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, group_json TEXT, receiver_json TEXT)")
    sources, qa = [], None
    counters = {"external_metric_hits": 0, "external_transport_hits": 0, "local_transport_hits": 0, "new_qa_calls": 0}
    try:
        completed = {r[0] for r in db.execute("SELECT id FROM cases")}
        if len(completed) == plan["expected_cases"]:
            export(config, plan, db, counters)
            return
        from bsrt_neural_compression.reward import QAReward
        qa, channels = QAReward(reference), Channels(config)
        schema = digest([qa.settings, (ROOT/"main.py").read_bytes().hex(), "all-MiniLM-L6-v2", False])
        for folder, kind in [("multicast_coding_budget/results/compressed40", "compressed"),
                             ("multicast_coding_budget/results/full", "original"),
                             ("semantic_multicast/results/heterogeneous_equal_budget", "legacy")]:
            parent = resolve(folder)
            if not (parent/"results.sqlite3").exists():
                continue
            lock = read_json(parent/"run_lock.json")
            physics = ["ldpc_iterations", "tdl_model", "delay_spread_seconds", "carrier_frequency_hz", "bandwidth_hz", "l_min", "l_max"]
            safe = all(lock["config"].get(p) == config[p] for p in physics)
            source_hashes = lock.get("implementation", {})
            if kind == "compressed":
                ancestor = read_json(resolve("multicast_coding_budget/results/full/run_lock.json"))
                safe = safe and lock["original_run_lock_sha256"] == digest(resolve("multicast_coding_budget/results/full/run_lock.json").read_bytes())
                source_hashes = {**ancestor["implementation"], **source_hashes}
            safe = safe and all(source_hashes.get(p) == digest((ROOT/p).read_bytes()) for p in [
                "bsrt_communication_evaluation/channel.py", "bsrt_budgeted_ldpc/channel.py"])
            source = sqlite3.connect((parent/"results.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)
            sources.append((source, kind, safe))

        def metric(text, q):
            key = digest([schema, text, q["question"], q["answers"]["text"]])
            saved = db.execute("SELECT value FROM metrics WHERE id=?", (key,)).fetchone()
            if saved:
                return json.loads(saved[0])
            for source, _, _ in sources:
                saved = source.execute("SELECT value FROM metrics WHERE id=?", (key,)).fetchone()
                if saved:
                    value = {f: json.loads(saved[0])[f] for f in FIELDS}
                    counters["external_metric_hits"] += 1
                    break
            else:
                row = qa.metrics(text, [q], allow_no_answer=False, semantic=True)[0]
                value = {f: row[f] for f in FIELDS}
            db.execute("INSERT INTO metrics VALUES (?,?)", (key, json.dumps(value)))
            return value

        def transport(frame, channel, snr, seed):
            key = digest(["task500_v1", frame.fingerprint, frame.coding, frame.blocks, len(frame.information), channel, snr, seed])
            saved = db.execute("SELECT value FROM transports WHERE id=?", (key,)).fetchone()
            if saved:
                counters["local_transport_hits"] += 1
                return json.loads(saved[0])
            for source, kind, safe in sources:
                if not safe:
                    continue
                lookup = None
                if kind == "compressed" and frame.coding == "ldpc":
                    lookup = digest(["compressed40_budget_v1", frame.fingerprint, frame.blocks, channel, snr, seed])
                elif kind == "original" and frame.coding == "uncoded":
                    lookup = digest(["original_segmented_v1", frame.fingerprint, "uncoded", (), len(frame.information), channel, snr, seed])
                elif frame.coding == "ldpc" and all(n == wire for _, n, wire in frame.blocks):
                    blocks = tuple((k, n) for k, n, _ in frame.blocks)
                    if kind == "original":
                        lookup = digest(["original_segmented_v1", frame.fingerprint, "ldpc", blocks, len(frame.information), channel, snr, seed])
                    elif kind == "legacy" and len(blocks) == 1:
                        lookup = digest([frame.fingerprint, "ldpc", len(frame.information), channel, snr, seed])
                if lookup:
                    saved = source.execute("SELECT value FROM transports WHERE id=?", (lookup,)).fetchone()
                    if saved:
                        value = json.loads(saved[0])
                        counters["external_transport_hits"] += 1
                        db.execute("INSERT INTO transports VALUES (?,?)", (key, json.dumps(value)))
                        return value
            decoded, stats = channels.receive(frame, channel, snr, seed)
            _, usable, invalid = received_ids(decoded, qa.tokenizer)
            value = {"stats": {**stats, **invalid}, "text": decode_ids(usable, qa.tokenizer)}
            db.execute("INSERT INTO transports VALUES (?,?)", (key, json.dumps(value)))
            return value

        conditions = list(product(config["channels"], config["snr_db_values"], config["channel_seeds"], ["unicast", "multicast"]))
        start = time.monotonic()
        for gi, g in enumerate(groups, 1):
            def identity(c, snr, seed, scenario):
                return {"context_id": g["context_id"], "channel": c, "snr_db": float(snr), "channel_seed": seed, "scenario": scenario}
            pending = [cond for cond in conditions if digest(identity(*cond)) not in completed]
            if not pending:
                continue
            unit = budget_for_context(g["original_tokens"])["per_receiver_budget_bits"]
            unicast = [channels.make_frame(ids, g["original_tokens"], 1) for ids in g["individual_ids"]]
            cluster_frames = {c["id"]: channels.make_frame(c["ids"], g["original_tokens"], len(c["members"])) for c in g["clusters"]}
            assignments = {i: c["id"] for c in g["clusters"] for i in c["members"]}
            multicast = [cluster_frames[assignments[i]] for i in range(g["k"])]
            offsets = np.linspace(-config["heterogeneous_span_db"]/2, config["heterogeneous_span_db"]/2, g["k"])
            for scenario, frames in [("unicast", unicast), ("multicast", multicast)]:
                with db:
                    for i, q in enumerate(g["questions"]):
                        key = digest([g["context_id"], q["id"], scenario])
                        if db.execute("SELECT 1 FROM clean WHERE id=?", (key,)).fetchone():
                            continue
                        text = decode_ids(g["individual_ids"][i] if scenario == "unicast" else
                                          next(c["ids"] for c in g["clusters"] if c["id"] == assignments[i]), qa.tokenizer)
                        value = {"context_id": g["context_id"], "receiver_id": q["id"], "scenario": scenario,
                                 "received_context": text, **metric(text, q)}
                        db.execute("INSERT INTO clean VALUES (?,?)", (key, json.dumps(value)))
            for channel, snr, seed, scenario in pending:
                record = identity(channel, snr, seed, scenario)
                frames = unicast if scenario == "unicast" else multicast
                receivers = []
                with db:
                    for i, q in enumerate(g["questions"]):
                        frame = frames[i]
                        actual_snr = float(snr+offsets[i])
                        physical_seed = stable_seed(g["context_id"], q["id"], channel, float(snr), "heterogeneous", seed)
                        result = transport(frame, channel, actual_snr, physical_seed)
                        outcome = metric(result["text"], q)
                        receivers.append({"receiver_id": q["id"], "question": q["question"], "references": q["answers"]["text"],
                            "cluster_id": assignments[i] if scenario == "multicast" else i,
                            "receiver_snr_db": actual_snr, "physical_seed": physical_seed,
                            "waveform_sha256": frame.fingerprint, "coding": frame.coding,
                            "payload_bits": frame.payload_bits, "frame_transmitted_bits": frame.transmitted_bits,
                            "received_context": result["text"], **result["stats"], **outcome})
                    sent_frames = unicast if scenario == "unicast" else list(cluster_frames.values())
                    transmitted = sum(f.transmitted_bits for f in sent_frames)
                    if transmitted != g["k"]*unit:
                        raise AssertionError("Both scenarios must exactly meet the same 60% aggregate budget")
                    aggregate = {**record, "receiver_count": g["k"], "cluster_count": len(g["clusters"]),
                        "transmission_count": len(sent_frames), "original_tokens": g["original_tokens"],
                        "kept_tokens": budget_for_context(g["original_tokens"])["kept_tokens"],
                        "budget_bits": g["k"]*unit, "transmitted_bits": transmitted,
                        **{f"mean_{m}": float(np.mean([r[m] for r in receivers])) for m in METRICS}}
                    key = digest(record)
                    db.execute("INSERT INTO cases VALUES (?,?,?)", (key, json.dumps(aggregate), json.dumps(receivers)))
                completed.add(key)
                counters["new_qa_calls"] = qa.calls
                if len(completed)%5 == 0:
                    print(f"Saved {len(completed)}/{plan['expected_cases']} cases; context {gi}/{len(groups)} "
                          f"K={g['k']} C={len(g['clusters'])} {channel} SNR={snr} {scenario}; QA={qa.calls}; "
                          f"cached metrics={counters['external_metric_hits']}; elapsed={time.monotonic()-start:.0f}s", flush=True)
                    write_json(output/"status.json", {"saved_cases": len(completed), "expected_cases": plan["expected_cases"],
                                                      "complete": False, "current_invocation": counters})
            export(config, plan, db, counters)
        export(config, plan, db, counters)
    finally:
        if qa is not None:
            qa.close()
        for source, _, _ in sources:
            source.close()
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="task_aligned_multicast_500/config.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    if args.smoke:
        config.update(output_dir="task_aligned_multicast_500/results/smoke", snr_db_values=[0, 3], channel_seeds=[42])
    run(config, args.smoke, args.dry_run)
    if not args.dry_run:
        from .audit import audit
        from .plot import plot
        audit(config["output_dir"])
        plot(config["output_dir"])


if __name__ == "__main__":
    main()
