import argparse
import json
import sqlite3

from bsrt_neural_compression.common import read_json, resolve, write_json
from .transport import budget_for_context


def audit(output_dir):
    output = resolve(output_dir)
    plan, lock = read_json(output/"plan.json"), read_json(output/"run_lock.json")
    if not read_json(output/"status.json")["complete"]:
        raise ValueError("Run is incomplete")
    groups = {g["context_id"]: g for g in lock["groups"]}
    with sqlite3.connect(output/"results.sqlite3") as db:
        rows = db.execute("SELECT group_json, receiver_json FROM cases").fetchall()
        clean = {(v["context_id"], v["receiver_id"], v["scenario"]): v
                 for (raw,) in db.execute("SELECT value FROM clean") for v in [json.loads(raw)]}
    assert len(rows) == plan["expected_cases"]
    assert len(clean) == 2*plan["questions"]
    pairs = {}
    outcomes = 0
    for gj, rj in rows:
        g, rr = json.loads(gj), json.loads(rj)
        definition = groups[g["context_id"]]
        k = definition["k"]
        unit = budget_for_context(definition["original_tokens"])
        assert len(rr) == k and len({r["receiver_id"] for r in rr}) == k
        assert g["transmitted_bits"] == g["budget_bits"] == k*unit["per_receiver_budget_bits"]
        assert 10*g["budget_bits"] <= 6*k*unit["original_frame_bits"]
        assert g["mean_accuracy"] == sum(r["accuracy"] for r in rr)/k
        outcomes += k
        if g["scenario"] == "unicast":
            assert g["transmission_count"] == k
            assert all(r["coding"] == "uncoded" for r in rr)
        else:
            assert g["transmission_count"] == len(definition["clusters"])
            for cluster in definition["clusters"]:
                cluster_rows = [r for r in rr if r["cluster_id"] == cluster["id"]]
                assert len(cluster_rows) == len(cluster["members"])
                assert len({r["waveform_sha256"] for r in cluster_rows}) == 1
                assert all(r["frame_transmitted_bits"] == len(cluster_rows)*unit["per_receiver_budget_bits"] for r in cluster_rows)
                assert all(r["coding"] == ("uncoded" if len(cluster_rows) == 1 else "ldpc") for r in cluster_rows)
        for r in rr:
            assert r["payload_bits"] == 15*unit["kept_tokens"]
            assert r["information_ber"] == r["information_errors"]/r["payload_bits"]
            if r["information_errors"] == 0:
                baseline = clean[g["context_id"], r["receiver_id"], g["scenario"]]
                assert r["received_context"] == baseline["received_context"]
                assert r["accuracy"] == baseline["accuracy"]
        pair_key = (g["context_id"], g["channel"], g["snr_db"], g["channel_seed"])
        pairs.setdefault(pair_key, {})[g["scenario"]] = rr
    for pair in pairs.values():
        assert set(pair) == {"unicast", "multicast"}
        assert [(r["receiver_id"], r["receiver_snr_db"], r["physical_seed"]) for r in pair["unicast"]] == [
            (r["receiver_id"], r["receiver_snr_db"], r["physical_seed"]) for r in pair["multicast"]]
        for u, m in zip(pair["unicast"], pair["multicast"], strict=True):
            if m["coding"] == "uncoded":
                assert u["waveform_sha256"] == m["waveform_sha256"]
                assert u["received_context"] == m["received_context"]
    assert outcomes == plan["expected_receiver_outcomes"]
    result = {"passed": True, "contexts": plan["contexts"], "questions": plan["questions"],
              "paired_conditions": len(pairs), "cases": len(rows), "receiver_outcomes": outcomes,
              "sixty_percent_budget_verified": True, "all_receivers_and_seeds_matched": True,
              "error_free_predictions_match_clean_compressed_baseline": True}
    write_json(output/"audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="task_aligned_multicast_500/results/full")
    audit(parser.parse_args().output)
