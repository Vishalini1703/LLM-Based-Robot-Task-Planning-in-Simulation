from __future__ import annotations

import json
import unittest

from robot_planner.llm import GroqPlanner, PlannerError
from robot_planner.world import world_from_mapping


def world():
    return world_from_mapping(
        {
            "locations": [
                {"id": "table", "kind": "surface", "openable": False, "is_open": True},
                {"id": "basket", "kind": "container", "openable": False, "is_open": True},
            ],
            "objects": [{"id": "apple", "location": "table", "portable": True}],
            "robot": {"location": "table", "held_object": None},
        }
    )


class GroqPlannerTests(unittest.TestCase):
    def test_parses_ready_plan_and_sends_world_contract(self) -> None:
        captured = {}

        def transport(url, api_key, payload, timeout):
            captured.update(payload)
            content = {
                "status": "ready",
                "plan": {
                    "plan_id": "remote-1",
                    "goal": "Put apple in basket",
                    "steps": [
                        {"action": "find", "arguments": {"object": "apple"}},
                        {"action": "pick", "arguments": {"object": "apple"}},
                        {"action": "navigate", "arguments": {"location": "basket"}},
                        {
                            "action": "place",
                            "arguments": {"object": "apple", "location": "basket"},
                        },
                    ],
                },
                "message": None,
            }
            return {
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

        result = GroqPlanner("test-key", transport=transport).plan(
            "Put the apple in the basket", world()
        )
        self.assertTrue(result.ready)
        self.assertEqual("remote-1", result.plan["plan_id"])
        self.assertEqual({"type": "json_object"}, captured["response_format"])
        self.assertIn("allowed_actions", captured["messages"][1]["content"])
        self.assertIn('"find"', captured["messages"][1]["content"])
        self.assertEqual(15, result.usage["total_tokens"])
        self.assertEqual(1, len(result.attempts))

    def test_parses_explicit_rejection(self) -> None:
        def transport(*_):
            content = {
                "status": "rejected",
                "plan": None,
                "message": "The banana is not in the declared kitchen.",
            }
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

        result = GroqPlanner("test-key", transport=transport).plan("Cook a banana", world())
        self.assertFalse(result.ready)
        self.assertIn("banana", result.message)

    def test_rejects_invalid_envelope(self) -> None:
        def transport(*_):
            return {"choices": [{"message": {"content": '{"plan": null}'}}]}

        with self.assertRaises(PlannerError):
            GroqPlanner("test-key", max_attempts=1, transport=transport).plan(
                "Move", world()
            )

    def test_retries_a_plan_rejected_by_local_validator(self) -> None:
        calls = []

        def transport(url, api_key, payload, timeout):
            calls.append(payload)
            if len(calls) == 1:
                plan = {
                    "plan_id": "bad-arguments",
                    "goal": "Put apple in basket",
                    "steps": [{"action": "pick", "arguments": ["apple"]}],
                }
            else:
                plan = {
                    "plan_id": "corrected",
                    "goal": "Put apple in basket",
                    "steps": [
                        {"action": "find", "arguments": {"object": "apple"}},
                        {"action": "pick", "arguments": {"object": "apple"}},
                        {"action": "navigate", "arguments": {"location": "basket"}},
                        {
                            "action": "place",
                            "arguments": {"object": "apple", "location": "basket"},
                        },
                    ],
                }
            content = {"status": "ready", "plan": plan, "message": None}
            return {"choices": [{"message": {"content": json.dumps(content)}}]}

        result = GroqPlanner("test-key", max_attempts=2, transport=transport).plan(
            "Move apple", world()
        )
        self.assertEqual("corrected", result.plan["plan_id"])
        self.assertEqual(2, len(calls))
        self.assertEqual(4, len(calls[1]["messages"]))
        self.assertEqual(2, len(result.attempts))
        self.assertEqual("validation_rejected", result.attempts[0].status)
        self.assertTrue(result.correction_succeeded)


if __name__ == "__main__":
    unittest.main()
