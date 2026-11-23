"""Check exact baseline reuse, paired receivers and aggregate bit budgets."""
import argparse
import json
import sqlite3

from bsrt_neural_compression.common import read_json, resolve
from .compressed_transport import compressed_allocation


def audit(output_dir):
    output = resolve(output_dir)
    status, plan = read_json(output/"status.json"), read_json(output/"plan.json")
    if not status["complete"]:
        raise ValueError("Evaluation incomplete")
    with sqlite3.connect(output/"results.sqlite3") as db:
        rows = db.execute("SELECT id, group_json, receiver_json FROM cases").fetchall()
    with sqlite3.connect(resolve("multicast_coding_budget/results/full/results.sqlite3")) as original:
        pairs = {}
        receiver_count = original_count = 0
        for key, gj, rj in rows:
            g, receivers = json.loads(gj), json.loads(rj)
            k = g["receiver_count"]
            assert len(receivers) == k
            assert len({r["receiver_id"] for r in receivers}) == k
            assert len({r["physical_seed"] for r in receivers}) == k
            assert g["total_transmitted_bits"] == g["budget_bits"]
            assert g["mean_accuracy"] == sum(r["accuracy"] for r in receivers)/k
            assert g["all_receivers_decode"] == int(all(r["decode_success"] for r in receivers))
            receiver_count += k
            pair = (g["group_id"], g["channel"], g["snr_db"], g["channel_seed"])
            pairs.setdefault(pair, {})[g["scenario"]] = (g, receivers)
            if g["scenario"] == "unicast":
                saved = original.execute("SELECT group_json, receiver_json FROM cases WHERE id=?", (key,)).fetchone()
                assert saved == (gj, rj), "Original baseline must be imported verbatim"
                assert g["removal_percent"] == 0 and g["transmission_count"] == k
                original_count += 1
            else:
                p = compressed_allocation(g["source_token_count"], g["kept_token_count"], k)
                assert g["removal_percent"] == 40 and g["transmission_count"] == 1
                assert g["kept_token_count"] == g["source_token_count"]-int(g["source_token_count"]*.4)
                assert p["budget_bits"] == g["budget_bits"]
                assert p["repeated_bits"] == g["repeated_bits"]
                assert len({r["waveform_sha256"] for r in receivers}) == 1
                for r in receivers:
                    assert r["payload_bits"] == 15*g["kept_token_count"]
                    assert r["transmitted_bits"] == g["budget_bits"]
                    assert r["information_ber"] == r["information_errors"]/r["payload_bits"]
        for pair in pairs.values():
            assert set(pair) == {"unicast", "multicast"}
            u, ur = pair["unicast"]
            m, mr = pair["multicast"]
            assert u["budget_bits"] == m["budget_bits"]
            assert [(r["receiver_id"], r["physical_seed"], r["receiver_snr_db"]) for r in ur] == [
                (r["receiver_id"], r["physical_seed"], r["receiver_snr_db"]) for r in mr]
    assert len(rows) == plan["expected_cases"]
    assert receiver_count == plan["expected_receiver_outcomes"]
    result = {"cases": len(rows), "receiver_outcomes": receiver_count,
              "original_cases_reused_verbatim": original_count, "paired_comparisons": len(pairs),
              "equal_aggregate_budget": True, "matched_receiver_seeds": True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="multicast_coding_budget/results/compressed40")
    audit(parser.parse_args().output)
