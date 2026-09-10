from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robot_planner.evaluation import (
    EvaluationCase,
    analyse_evaluation,
    assess_trial,
    freeze_settings,
    load_catalog,
    state_matches,
    verify_frozen_settings,
)


class EvaluationTests(unittest.TestCase):
    def test_frozen_catalog_has_required_balance(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cases = load_catalog(project_root / "evaluation" / "commands.json")
        self.assertEqual(30, len(cases))
        self.assertEqual(6, sum(case.category == "valid_single" for case in cases))
        self.assertEqual(8, sum(case.category == "valid_multi" for case in cases))
        self.assertEqual(
            6, sum(case.category == "invalid_precondition" for case in cases)
        )
        self.assertEqual(5, sum(case.category == "unsupported" for case in cases))
        self.assertEqual(5, sum(case.category == "ambiguous" for case in cases))

    def test_state_match_checks_only_frozen_expected_subset(self) -> None:
        actual = {
            "objects": [{"id": "apple", "location": "basket", "portable": True}],
            "locations": [
                {
                    "id": "cupboard",
                    "kind": "container",
                    "openable": True,
                    "is_open": False,
                }
            ],
            "robot": {
                "location": "basket",
                "held_object": None,
                "observed_objects": ["apple"],
            },
        }
        expected = {
            "object_locations": {"apple": "basket"},
            "location_open": {"cupboard": False},
            "robot": {"location": "basket", "held_object": None},
        }
        self.assertTrue(state_matches(actual, expected))
        expected["object_locations"]["apple"] = "table"
        self.assertFalse(state_matches(actual, expected))

    def test_assesses_correct_rejection(self) -> None:
        case = EvaluationCase(
            id="T",
            category="unsupported",
            command="Cook apple",
            expected_executable=False,
            expected_statuses=("planner_rejected", "validation_rejected"),
            expected_issue_codes=("UNKNOWN_ACTION",),
            expected_state={},
        )
        assessment = assess_trial(
            case,
            {
                "status": "planner_rejected",
                "validation": None,
                "failure": None,
                "execution": None,
            },
        )
        self.assertTrue(assessment["correct_rejection"])
        self.assertTrue(assessment["classification_correct"])

    def test_freeze_detects_changed_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("one", encoding="utf-8")
            protocol = root / "protocol.json"
            protocol.write_text(
                json.dumps(
                    {
                        "version": "1",
                        "model": "test",
                        "temperature": 0,
                        "max_completion_tokens": 10,
                        "max_attempts": 2,
                        "timeout_seconds": 1,
                        "repetitions": 5,
                        "backend": "memory",
                        "frozen_files": ["input.txt"],
                    }
                ),
                encoding="utf-8",
            )
            frozen_path = root / "frozen.json"
            freeze_settings(root, protocol, frozen_path)
            verify_frozen_settings(root, frozen_path)
            (root / "input.txt").write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Frozen evaluation inputs changed"):
                verify_frozen_settings(root, frozen_path)

    def test_analysis_calculates_declared_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "trials"
            trials.mkdir()
            valid = {
                "trial_id": "V-r01",
                "repetition": 1,
                "case": {
                    "id": "V",
                    "category": "valid_single",
                    "expected_executable": True,
                },
                "result": {
                    "status": "executed",
                    "duration_ms": 12.0,
                    "planner": {
                        "status": "ready",
                        "plan": {
                            "steps": [
                                {"action": "navigate", "arguments": {"location": "table"}}
                            ]
                        },
                        "attempt_count": 1,
                        "total_latency_ms": 10.0,
                        "correction_attempted": False,
                        "correction_succeeded": False,
                    },
                    "validation": {"valid": True, "issues": []},
                },
                "assessment": {
                    "observed_issue_code": None,
                    "classification_correct": True,
                    "task_success": True,
                    "correct_rejection": False,
                },
            }
            rejected = {
                "trial_id": "U-r01",
                "repetition": 1,
                "case": {
                    "id": "U",
                    "category": "unsupported",
                    "expected_executable": False,
                },
                "result": {
                    "status": "planner_rejected",
                    "duration_ms": 8.0,
                    "planner": {
                        "status": "rejected",
                        "plan": None,
                        "attempt_count": 1,
                        "total_latency_ms": 7.0,
                        "correction_attempted": False,
                        "correction_succeeded": False,
                    },
                    "validation": None,
                },
                "assessment": {
                    "observed_issue_code": None,
                    "classification_correct": True,
                    "task_success": False,
                    "correct_rejection": True,
                },
            }
            (trials / "V-r01.json").write_text(
                json.dumps(valid), encoding="utf-8"
            )
            (trials / "U-r01.json").write_text(
                json.dumps(rejected), encoding="utf-8"
            )
            metrics = analyse_evaluation(trials, root / "analysis")
            self.assertEqual(100.0, metrics["structured_response"]["percent"])
            self.assertEqual(100.0, metrics["plan_validity"]["percent"])
            self.assertEqual(100.0, metrics["task_success"]["percent"])
            self.assertEqual(100.0, metrics["correct_rejection"]["percent"])
            self.assertTrue((root / "analysis" / "trials.csv").is_file())
            self.assertTrue((root / "analysis" / "audit-sample.json").is_file())


if __name__ == "__main__":
    unittest.main()
