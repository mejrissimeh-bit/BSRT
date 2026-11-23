"""Import the audited matching smoke cases/caches before the full run starts."""
import json
import sqlite3

from bsrt_neural_compression.common import read_json, resolve, write_json


def main():
    smoke = resolve("task_aligned_multicast_500/results/smoke")
    output = resolve("task_aligned_multicast_500/results/full")
    source_lock, target_lock = read_json(smoke/"run_lock.json"), read_json(output/"run_lock.json")
    if not read_json(smoke/"audit.json")["passed"]:
        raise ValueError("Only audited results can be reused")
    for key in ["implementation", "software", "qa_config_sha256"]:
        if source_lock[key] != target_lock[key]:
            raise ValueError(f"Different {key}")
    for key, value in source_lock["config"].items():
        if key in ["output_dir", "snr_db_values", "channel_seeds"]:
            continue
        if target_lock["config"].get(key) != value:
            raise ValueError(f"Different configuration: {key}")
    if not all(g in target_lock["groups"] for g in source_lock["groups"]):
        raise ValueError("Payloads, questions or clusters differ")
    for field in ["snr_db_values", "channel_seeds"]:
        if not set(source_lock["config"][field]) <= set(target_lock["config"][field]):
            raise ValueError("Smoke conditions are outside the full run")
    counts = {}
    with sqlite3.connect(smoke/"results.sqlite3") as source, sqlite3.connect(output/"results.sqlite3") as target:
        target.execute("PRAGMA journal_mode=WAL")
        for table in ["cases", "clean", "metrics", "transports"]:
            ddl = source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
            if not target.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                target.execute(ddl)
            rows = source.execute(f"SELECT * FROM {table}").fetchall()
            for row in rows:
                saved = target.execute(f"SELECT * FROM {table} WHERE id=?", (row[0],)).fetchone()
                if saved is not None and saved != row:
                    raise ValueError(f"Conflicting cache value in {table}")
                if saved is None:
                    target.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in row)})", row)
            counts[table] = len(rows)
    record = {"source": str(smoke), "validated_imports": counts,
              "same_payload_questions_clusters_physics_and_software": True}
    write_json(output/"smoke_cache_import.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
