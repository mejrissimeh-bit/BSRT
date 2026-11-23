"""Validate saved cases before admitting the zero-pruning decoder fix."""
import json
import sqlite3

from bsrt_neural_compression.common import ROOT, digest, read_json, resolve, write_json
from .transport import OriginalChannels


def main():
    output = resolve("multicast_coding_budget/results/full")
    backup = output/"decoder_fix_backup"
    lock = read_json(output/"run_lock.json")
    old = read_json(backup/"run_lock.json")
    transport = "multicast_coding_budget/transport.py"
    if lock != old:
        raise ValueError("Run lock already changed; inspect migration record")
    if digest((backup/"transport.py").read_bytes()) != old["implementation"][transport]:
        raise ValueError("Archived source differs from the original run")
    for path, expected in old["implementation"].items():
        if path != transport and digest((ROOT/path).read_bytes()) != expected:
            raise ValueError(f"Unrelated source change: {path}")

    import torch
    from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    torch.set_num_threads(2)
    channels = OriginalChannels(lock["config"])
    plans = read_json(output/"budget_allocations.json")
    affected = set()
    dimensions = sorted({(b["k"], b["n"]) for p in plans for b in p["blocks"]})
    for k, n in dimensions:
        encoder = LDPC5GEncoder(k, n, device="cpu")
        original = LDPC5GDecoder(encoder, num_iter=12, device="cpu")
        if original.num_vns == 0:
            affected.add((k, n))
        frame_bits = torch.randint(0, 2, (1, k)).float()
        enc, dec = channels.codec(k, n)
        with torch.inference_mode():
            codeword = enc(frame_bits)
            decoded = dec((2*codeword-1)*20)
        if not torch.equal(decoded, frame_bits):
            raise AssertionError(f"Noiseless decoding failed: {(k, n)}")
    affected_groups = {p["group_id"] for p in plans
                       if any((b["k"], b["n"]) in affected for b in p["blocks"])}
    with sqlite3.connect(output/"results.sqlite3") as db:
        completed = [json.loads(row[0]) for row in db.execute("SELECT group_json FROM cases")]
        if any(g["group_id"] in affected_groups and g["scenario"] == "multicast" for g in completed):
            raise ValueError("Affected decoder already has saved cases; do not reuse them automatically")
        with sqlite3.connect(backup/"results.sqlite3") as snapshot:
            db.backup(snapshot)
    lock["implementation"][transport] = digest((ROOT/transport).read_bytes())
    record = {"reason": "Sionna 2.0.1 zero-node pruning slices PCM with [:-0]",
              "fix": "Disable pruning only when default decoder has zero variable nodes",
              "old_source_hash": old["implementation"][transport],
              "new_source_hash": lock["implementation"][transport],
              "affected_dimensions": sorted(affected), "validated_dimensions": len(dimensions),
              "preserved_cases": len(completed), "affected_completed_multicast_cases": 0,
              "allocation_encoder_channel_seeds_and_metrics_changed": False}
    write_json(backup/"migration.json", record)
    write_json(output/"run_lock.json", lock)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
