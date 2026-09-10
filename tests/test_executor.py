from __future__ import annotations

import unittest

from robot_planner.executor import ExecutionRejected, InMemoryExecutor
from robot_planner.models import Plan
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


class ExecutorTests(unittest.TestCase):
    def test_executes_validated_plan_with_step_audit(self) -> None:
        plan = Plan.from_mapping(
            {
                "plan_id": "execute-1",
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
        )
        initial = world()
        result = InMemoryExecutor().execute(plan, initial)
        self.assertEqual(4, len(result.steps))
        self.assertTrue(all(step.outcome_verified for step in result.steps))
        final_objects = {item["id"]: item for item in result.final_state["objects"]}
        self.assertEqual("basket", final_objects["apple"]["location"])
        self.assertEqual("table", initial.objects["apple"].location)

    def test_blocks_invalid_plan_before_state_change(self) -> None:
        plan = Plan.from_mapping(
            {
                "plan_id": "execute-bad",
                "goal": "Place an object not held",
                "steps": [
                    {
                        "action": "place",
                        "arguments": {"object": "apple", "location": "table"},
                    }
                ],
            }
        )
        initial = world()
        with self.assertRaises(ExecutionRejected) as raised:
            InMemoryExecutor().execute(plan, initial)
        self.assertEqual("OBJECT_NOT_HELD", raised.exception.validation.issues[0].code)
        self.assertEqual("table", initial.objects["apple"].location)

    def test_stops_when_observed_state_differs_from_expected(self) -> None:
        plan = Plan.from_mapping(
            {
                "plan_id": "observe-bad",
                "goal": "Find apple",
                "steps": [{"action": "find", "arguments": {"object": "apple"}}],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "Observed state did not match"):
            InMemoryExecutor(observer=lambda _: {"wrong": "state"}).execute(
                plan,
                world(),
            )


if __name__ == "__main__":
    unittest.main()
