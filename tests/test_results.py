import csv
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
