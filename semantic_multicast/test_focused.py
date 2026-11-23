import copy
import json
import sqlite3
import unittest

from bsrt_neural_compression.common import digest
from .run_focused import AXES, KEY_FIELDS, compatible, copy_matching, insert_exact


class ImportTests(unittest.TestCase):
    def setUp(self):
        config = {"budget_regimes": ["equal_aggregate"], "snr_profiles": ["heterogeneous"],
                  "channels": ["awgn"], "coding_modes": ["ldpc"], "receiver_counts": [2],
                  "removal_percentages": [40], "snr_db_values": [3.0], "channel_seeds": [42],
                  "selection_seed": 42, "output_dir": "new", "max_contexts": None}
        group = {"group_id": "g", "context_id": "c", "receiver_count": 2, "receiver_ids": [1, 2]}
        self.target_lock = {"config": config, "groups": [group], "implementation": {"run.py": "same"},
                            "qa_config_sha256": "same", "software": {"sionna": "same"}}
        self.source_lock = copy.deepcopy(self.target_lock)
        self.source_lock["config"].update(output_dir="old", budget_regimes=["equal_aggregate", "equal_per_representation"],
                                            snr_profiles=["homogeneous", "heterogeneous"])

    def database(self):
        db = sqlite3.connect(":memory:")
        for table, columns in [("cases", "group_json TEXT, receiver_json TEXT"), ("metrics", "value TEXT"), ("transports", "value TEXT")]:
            db.execute(f"CREATE TABLE {table} (id TEXT PRIMARY KEY, {columns})")
        self.addCleanup(db.close)
        return db

    def add_case(self, db, regime, profile):
        record = {field: self.target_lock["config"][key][0] for key, field in AXES.items()}
        record.update(group_id="g", context_id="c", budget_regime=regime, snr_profile=profile,
                      scenario="multicast", total_transmitted_bits=100, aggregate_ceiling_bits=100)
        key = digest({field: record[field] for field in KEY_FIELDS})
        receivers = [{"receiver_id": "1"}, {"receiver_id": "2"}]
        db.execute("INSERT INTO cases VALUES (?,?,?)", (key, json.dumps(record), json.dumps(receivers)))

    def test_filter_and_idempotent_exact_reuse(self):
        source, target = self.database(), self.database()
        for regime in ["equal_aggregate", "equal_per_representation"]:
            for profile in ["homogeneous", "heterogeneous"]:
                self.add_case(source, regime, profile)
        source.execute("INSERT INTO metrics VALUES ('m','metric')")
        source.execute("INSERT INTO transports VALUES ('t','transport')")
        first = copy_matching(source, target, self.source_lock, self.target_lock)
        self.assertEqual(first["imported_cases"], 1)
        self.assertEqual(first["imported_receiver_outcomes"], 2)
        self.assertEqual(first["imported_metrics"], 1)
        self.assertEqual(first["imported_transports"], 1)
        second = copy_matching(source, target, self.source_lock, self.target_lock)
        self.assertEqual(second["imported_cases"], 0)
        self.assertEqual(second["imported_metrics"], 0)
        self.assertEqual(source.execute("SELECT COUNT(*) FROM cases").fetchone()[0], 4)

    def test_changed_physics_or_implementation_rejected(self):
        for part, field in [("config", "selection_seed"), ("implementation", "run.py"), ("software", "sionna")]:
            target = copy.deepcopy(self.target_lock)
            target[part][field] = "different"
            with self.assertRaises(ValueError):
                compatible(self.source_lock, target)

    def test_changed_group_rejected(self):
        target = copy.deepcopy(self.target_lock)
        target["groups"][0]["receiver_ids"] = [2, 3]
        with self.assertRaises(ValueError):
            compatible(self.source_lock, target)

    def test_partial_pilot_grid_can_be_reused(self):
        source = copy.deepcopy(self.source_lock)
        target = copy.deepcopy(self.target_lock)
        target["config"]["snr_db_values"] = [0, 3, 5, 7, 15]
        target["groups"].append({"group_id": "another", "receiver_ids": [3, 4], "receiver_count": 2})
        compatible(source, target)

    def test_conflicting_cache_not_overwritten(self):
        target = self.database()
        insert_exact(target, "metrics", "m", ("old",))
        with self.assertRaises(ValueError):
            insert_exact(target, "metrics", "m", ("different",))


if __name__ == "__main__":
    unittest.main()
