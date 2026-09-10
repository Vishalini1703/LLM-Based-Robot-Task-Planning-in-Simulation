from __future__ import annotations

import unittest

from robot_planner.llm import PlannerResponse
from robot_planner.pipeline import TaskPlanningPipeline
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


class FakePlanner:
    def __init__(self, response):
        self.response = response

    def plan(self, command, initial_world):
        return self.response


class PipelineTests(unittest.TestCase):
    def test_executes_ready_valid_plan(self) -> None:
        response = PlannerResponse(
            status="ready",
            model="fake",
            latency_ms=1,
            raw_response="{}",
            plan={
                "plan_id": "pipeline-1",
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
            message=None,
            usage={},
        )
        result = TaskPlanningPipeline(FakePlanner(response)).run("move apple", world())
        self.assertEqual("executed", result.status)
        self.assertIsNotNone(result.execution)

    def test_stops_on_planner_rejection(self) -> None:
        response = PlannerResponse(
            status="rejected",
            model="fake",
            latency_ms=1,
            raw_response="{}",
            plan=None,
            message="Unsupported",
            usage={},
        )
        result = TaskPlanningPipeline(FakePlanner(response)).run("cook", world())
        self.assertEqual("planner_rejected", result.status)
        self.assertIsNone(result.validation)
        self.assertIsNone(result.execution)

    def test_stops_on_validator_rejection(self) -> None:
        response = PlannerResponse(
            status="ready",
            model="fake",
            latency_ms=1,
            raw_response="{}",
            plan={
                "plan_id": "pipeline-bad",
                "goal": "Invent action",
                "steps": [{"action": "cook", "arguments": {"object": "apple"}}],
            },
            message=None,
            usage={},
        )
        result = TaskPlanningPipeline(FakePlanner(response)).run("cook", world())
        self.assertEqual("validation_rejected", result.status)
        self.assertEqual("UNKNOWN_ACTION", result.validation.issues[0].code)


if __name__ == "__main__":
    unittest.main()
