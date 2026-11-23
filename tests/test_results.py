import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def rows_at(removal):
    with (ROOT / "results" / "comparison_summary.csv").open(newline="", encoding="utf-8") as handle:
        return {
            row["method"]: row
            for row in csv.DictReader(handle)
            if int(row["removal_percent"]) == removal
        }


def test_published_headline_metrics_match_archived_results():
    at_40 = rows_at(40)
    at_50 = rows_at(50)
    case = unittest.TestCase()
    case.assertAlmostEqual(float(at_40["bsrt"]["f1"]), 0.7497538216)
    case.assertAlmostEqual(float(at_40["l2"]["f1"]), 0.5798639086)
    case.assertAlmostEqual(float(at_40["bsrt"]["accuracy"]), 0.812)
    case.assertAlmostEqual(float(at_50["bsrt"]["f1"]), 0.6813442355)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_equal_budget_ldpc_headline_results_and_audit():
    rows = read_csv(ROOT / "results" / "ldpc_equal_budget" / "awgn" / "summary.csv")
    by_condition = {
        (int(row["removal_percent"]), float(row["channel_value"])): row
        for row in rows
    }
    case = unittest.TestCase()
    case.assertAlmostEqual(float(by_condition[(40, 3.0)]["accuracy"]), 0.815)
    case.assertAlmostEqual(float(by_condition[(30, 5.0)]["accuracy"]), 0.860)
    case.assertAlmostEqual(float(by_condition[(10, 7.0)]["accuracy"]), 0.9216666667)
    for row in rows:
        case.assertAlmostEqual(float(row["transmitted_bits"]), float(row["original_budget_bits"]))
    checks = json.loads(
        (ROOT / "results" / "ldpc_equal_budget" / "awgn" / "verification_checks.json").read_text(encoding="utf-8")
    )
    assert len(checks["checks"]) == 30
    assert all(checks["checks"].values())


def test_unicast_multicast_results_equal_budget_and_verification():
    rows = read_csv(ROOT / "results" / "unicast_multicast" / "summary_all_questions.csv")
    by_condition = {
        (row["channel"], float(row["snr_db"]), row["scenario"]): row
        for row in rows
    }
    case = unittest.TestCase()
    case.assertAlmostEqual(float(by_condition[("awgn", 3.0, "unicast")]["accuracy"]), 0.2315705128)
    case.assertAlmostEqual(float(by_condition[("awgn", 3.0, "multicast")]["accuracy"]), 0.8774038462)
    case.assertAlmostEqual(float(by_condition[("fading", 5.0, "multicast")]["accuracy"]), 0.9046474359)
    metadata = json.loads(
        (ROOT / "results" / "unicast_multicast" / "subset_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["questions"] == 416
    assert metadata["clusters"] == 109
    assert metadata["aggregate_budget_per_strategy_per_channel_snr_seed_bits"] == 524268
    checks = json.loads(
        (ROOT / "results" / "unicast_multicast" / "verification" / "checks.json").read_text(encoding="utf-8")
    )
    assert checks["passed"] is True
