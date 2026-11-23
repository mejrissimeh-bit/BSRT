"""Import verified matching results, then run only heterogeneous equal-budget cases."""
from __future__ import annotations

import argparse
import json
import sqlite3

from bsrt_neural_compression.common import digest, read_json, resolve, write_json
from .run import export_results, open_results, prepare, run


DEFAULT = "semantic_multicast/config_heterogeneous_equal_budget.json"
SOURCE = "semantic_multicast/results/full"
AXES = {"budget_regimes": "budget_regime", "snr_profiles": "snr_profile", "channels": "channel",
        "coding_modes": "coding", "receiver_counts": "receiver_count",
        "removal_percentages": "removal_percent", "snr_db_values": "snr_db", "channel_seeds": "channel_seed"}
KEY_FIELDS = ["group_id", "context_id", "receiver_count", "budget_regime", "removal_percent",
              "coding", "channel", "snr_profile", "snr_db", "channel_seed", "scenario"]


def compatible(source, target):
    for field in ["implementation", "qa_config_sha256", "software"]:
        if source[field] != target[field]:
            raise ValueError(f"Cannot reuse results: {field} differs")
    old, new = source["config"], target["config"]
    ignored = set(AXES) | {"output_dir", "max_contexts"}
    if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in new.items() if k not in ignored}:
        raise ValueError("Non-grid scientific settings differ; cached results are not interchangeable")
    for key in AXES:
        if not set(new[key]).intersection(old[key]):
            raise ValueError(f"No reusable overlap for {key}")
    old_groups = {g["group_id"]: g for g in source["groups"]}
    overlap = 0
    for group in target["groups"]:
        if group["group_id"] not in old_groups:
            continue
        overlap += 1
        if old_groups[group["group_id"]] != group:
            raise ValueError("Receiver group, cached gradients or source tokens changed")
    if not overlap:
        raise ValueError("No verified receiver groups in common")


def selected(record, config, groups):
    return record["group_id"] in groups and all(record[field] in config[key] for key, field in AXES.items())


def insert_exact(db, table, key, values):
    columns = {"cases": ("group_json", "receiver_json"), "metrics": ("value",), "transports": ("value",)}
    names = columns[table]
    existing = db.execute(f"SELECT {','.join(names)} FROM {table} WHERE id=?", (key,)).fetchone()
    if existing is not None:
        if tuple(values) != existing:
            raise ValueError(f"Conflicting saved {table} entry: {key}")
        return 0
    db.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in range(len(values)+1))})", (key, *values))
    return 1


def copy_matching(source_db, target_db, source_lock, target_lock):
    compatible(source_lock, target_lock)
    groups = {g["group_id"]: g for g in target_lock["groups"]}
    counts = {"source_cases": 0, "matching_source_cases": 0, "imported_cases": 0,
              "imported_receiver_outcomes": 0, "imported_metrics": 0, "imported_transports": 0}
    with target_db:
        for key, group_json, receiver_json in source_db.execute("SELECT id, group_json, receiver_json FROM cases"):
            counts["source_cases"] += 1
            record = json.loads(group_json)
            if not selected(record, target_lock["config"], groups):
                continue
            if key != digest({field: record[field] for field in KEY_FIELDS}):
                raise ValueError("Saved case identity does not match its hash")
            receivers = json.loads(receiver_json)
            expected = groups[record["group_id"]]
            if len(receivers) != expected["receiver_count"] or [r["receiver_id"] for r in receivers] != list(map(str, expected["receiver_ids"])):
                raise ValueError("Saved receiver identities differ from the target cohort")
            if record["total_transmitted_bits"] > record["aggregate_ceiling_bits"]:
                raise ValueError("Imported equal-aggregate case exceeds its budget")
            counts["matching_source_cases"] += 1
            added = insert_exact(target_db, "cases", key, (group_json, receiver_json))
            counts["imported_cases"] += added
            counts["imported_receiver_outcomes"] += added*len(receivers)
        # Keys include the exact waveform/seed or received text/question/metric schema.
        # Nonmatching-condition cache entries cannot turn into false hits.
        for table in ["metrics", "transports"]:
            for key, value in source_db.execute(f"SELECT id, value FROM {table}"):
                counts[f"imported_{table}"] += insert_exact(target_db, table, key, (value,))
    return counts


def import_saved(config, source_dir):
    if config["budget_regimes"] != ["equal_aggregate"] or config["snr_profiles"] != ["heterogeneous"]:
        raise ValueError("This entry point is restricted to heterogeneous, equal-aggregate-budget cases")
    _, plan = prepare(config)
    output, source_dir = resolve(config["output_dir"]), resolve(source_dir)
    if output.resolve() == source_dir.resolve():
        raise ValueError("Import target must not overwrite the source experiment")
    source_path = source_dir/"results.sqlite3"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = sqlite3.connect(source_path.resolve().as_uri()+"?mode=ro", uri=True)
    target = open_results(output)
    try:
        if source.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Source SQLite integrity check failed")
        counts = copy_matching(source, target, read_json(source_dir/"run_lock.json"), read_json(output/"run_lock.json"))
        saved = target.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        details = {"source": str(source_dir), "target": str(output), **counts,
                   "saved_target_cases": saved, "expected_cases": plan["expected_cases"],
                   "remaining_cases": plan["expected_cases"]-saved,
                   "source_database_modified": False, "channel_or_qa_calls_during_import": 0}
        write_json(output/f"import_manifest_{source_dir.name}.json", details)
        if saved:
            export_results(config, plan, target)
        print("VERIFIED REUSE\n"+json.dumps(details, indent=2), flush=True)
        return details
    finally:
        source.close()
        target.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--extra-sources", nargs="*", default=["semantic_multicast/results/pilot", "semantic_multicast/results/smoke"])
    parser.add_argument("--prepare-only", action="store_true", help="Import and verify, without running any new channels or QA")
    args = parser.parse_args()
    config = read_json(resolve(args.config))
    imports = [import_saved(config, args.source)]
    for source in args.extra_sources:
        if (resolve(source)/"results.sqlite3").exists():
            imports.append(import_saved(config, source))
    write_json(resolve(config["output_dir"])/"import_manifest.json", {
        "sources": imports, "imported_cases_this_invocation": sum(i["imported_cases"] for i in imports),
        "saved_target_cases": imports[-1]["saved_target_cases"],
        "expected_cases": imports[-1]["expected_cases"], "remaining_cases": imports[-1]["remaining_cases"],
        "channel_or_qa_calls_during_import": 0})
    if not args.prepare_only:
        run(config)


if __name__ == "__main__":
    main()
