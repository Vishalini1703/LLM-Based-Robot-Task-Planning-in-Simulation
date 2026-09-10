import csv
import tempfile
import unittest
from pathlib import Path

from robot_planner.feedback import (
    REQUIRED_FIELDS,
    load_feedback,
    summarise_feedback,
)


class FeedbackTests(unittest.TestCase):
    def write_csv(self, rows):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "responses.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_validates_and_aggregates_anonymous_feedback(self):
        rows = [
            {
                "participant_id": "P001",
                "clarity": 4,
                "usability": 5,
                "understandability": 4,
                "usefulness": 5,
                "rejection_confidence": 3,
                "clearest_feature": "Plan",
                "most_confusing_feature": "",
                "suggested_improvement": "Larger text",
            },
            {
                "participant_id": "P002",
                "clarity": 2,
                "usability": 3,
                "understandability": 4,
                "usefulness": 3,
                "rejection_confidence": 5,
                "clearest_feature": "Result",
                "most_confusing_feature": "Error code",
                "suggested_improvement": "",
            },
        ]
        loaded = load_feedback(self.write_csv(rows))
        summary = summarise_feedback(loaded, invited_count=4)
        self.assertEqual(summary["completed_count"], 2)
        self.assertEqual(summary["completion_rate_percent"], 50.0)
        self.assertEqual(summary["ratings"]["clarity"]["mean"], 3.0)
        self.assertNotIn("P001", str(summary))

    def test_rejects_duplicate_participant_id(self):
        row = {
            "participant_id": "P001",
            **{field: 3 for field in REQUIRED_FIELDS[1:6]},
            **{field: "" for field in REQUIRED_FIELDS[6:]},
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_feedback(self.write_csv([row, row]))

    def test_rejects_out_of_range_rating(self):
        row = {
            "participant_id": "P001",
            **{field: 3 for field in REQUIRED_FIELDS[1:6]},
            **{field: "" for field in REQUIRED_FIELDS[6:]},
        }
        row["clarity"] = 6
        with self.assertRaisesRegex(ValueError, "clarity"):
            load_feedback(self.write_csv([row]))

    def test_rejects_impossible_invited_count(self):
        rows = [
            {
                "participant_id": "P001",
                **{field: 3 for field in REQUIRED_FIELDS[1:6]},
                **{field: "" for field in REQUIRED_FIELDS[6:]},
            }
        ]
        with self.assertRaisesRegex(ValueError, "lower"):
            summarise_feedback(rows, invited_count=0)
