from __future__ import annotations

import unittest

from robot_planner.validator import PlanValidator
from robot_planner.world import world_from_mapping


def test_world():
    return world_from_mapping(
        {
            "locations": [
                {"id": "table", "kind": "surface", "openable": False, "is_open": True},
                {"id": "basket", "kind": "container", "openable": False, "is_open": True},
                {"id": "cupboard", "kind": "container", "openable": True, "is_open": False},
            ],
            "objects": [
                {"id": "apple", "location": "table", "portable": True},
                {"id": "mug", "location": "table", "portable": True},
                {"id": "kettle", "location": "table", "portable": False},
            ],
            "robot": {"location": "table", "held_object": None},
        }
    )


def plan(*steps):
    return {"plan_id": "test-plan", "goal": "Test goal", "steps": list(steps)}


def step(action: str, **arguments: str):
    return {"action": action, "arguments": arguments}


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = PlanValidator()

    def test_valid_pick_navigate_place_sequence(self) -> None:
        world = test_world()
        result = self.validator.validate_mapping(
            plan(
                step("find", object="apple"),
                step("pick", object="apple"),
                step("navigate", location="basket"),
                step("place", object="apple", location="basket"),
            ),
            world,
        )
        self.assertTrue(result.valid)
        self.assertEqual(4, result.steps_checked)
        final_objects = {item["id"]: item for item in result.predicted_final_state["objects"]}
        self.assertEqual("basket", final_objects["apple"]["location"])
        self.assertIsNone(result.predicted_final_state["robot"]["held_object"])
        self.assertEqual("table", world.objects["apple"].location, "validation must not mutate input")

    def test_valid_open_place_close_sequence(self) -> None:
        result = self.validator.validate_mapping(
            plan(
                step("find", object="apple"),
                step("pick", object="apple"),
                step("navigate", location="cupboard"),
                step("open", container="cupboard"),
                step("place", object="apple", location="cupboard"),
                step("close", container="cupboard"),
            ),
            test_world(),
        )
        self.assertTrue(result.valid)
        final_locations = {
            location["id"]: location for location in result.predicted_final_state["locations"]
        }
        self.assertFalse(final_locations["cupboard"]["is_open"])

    def test_rejects_missing_required_plan_field(self) -> None:
        result = self.validator.validate_mapping({"goal": "No id", "steps": []}, test_world())
        self.assertFalse(result.valid)
        self.assertEqual("INVALID_PLAN_SCHEMA", result.issues[0].code)

    def test_rejects_unexpected_plan_field(self) -> None:
        raw = plan(step("pick", object="apple"))
        raw["reasoning"] = "not accepted"
        result = self.validator.validate_mapping(raw, test_world())
        self.assertEqual("INVALID_PLAN_SCHEMA", result.issues[0].code)

    def test_rejects_hallucinated_action(self) -> None:
        result = self.validator.validate_mapping(plan(step("cook", object="apple")), test_world())
        self.assertEqual("UNKNOWN_ACTION", result.issues[0].code)
        self.assertEqual(0, result.issues[0].step_index)

    def test_rejects_hallucinated_object(self) -> None:
        result = self.validator.validate_mapping(plan(step("pick", object="banana")), test_world())
        self.assertEqual("UNKNOWN_OBJECT", result.issues[0].code)

    def test_rejects_wrong_action_arguments(self) -> None:
        result = self.validator.validate_mapping(
            plan(step("navigate", location="basket", speed="fast")), test_world()
        )
        self.assertEqual("INVALID_ARGUMENTS", result.issues[0].code)

    def test_rejects_pick_when_robot_is_elsewhere(self) -> None:
        result = self.validator.validate_mapping(
            plan(step("navigate", location="basket"), step("pick", object="apple")), test_world()
        )
        self.assertEqual("ROBOT_NOT_AT_LOCATION", result.issues[0].code)
        self.assertEqual(1, result.issues[0].step_index)

    def test_rejects_second_pick_when_hand_is_occupied(self) -> None:
        result = self.validator.validate_mapping(
            plan(
                step("find", object="apple"),
                step("pick", object="apple"),
                step("find", object="mug"),
                step("pick", object="mug"),
            ),
            test_world(),
        )
        self.assertEqual("HAND_OCCUPIED", result.issues[0].code)

    def test_rejects_nonportable_object(self) -> None:
        result = self.validator.validate_mapping(plan(step("pick", object="kettle")), test_world())
        self.assertEqual("OBJECT_NOT_PORTABLE", result.issues[0].code)

    def test_rejects_place_in_closed_container(self) -> None:
        result = self.validator.validate_mapping(
            plan(
                step("find", object="apple"),
                step("pick", object="apple"),
                step("navigate", location="cupboard"),
                step("place", object="apple", location="cupboard"),
            ),
            test_world(),
        )
        self.assertEqual("CONTAINER_CLOSED", result.issues[0].code)
        self.assertEqual(3, result.issues[0].step_index)

    def test_rejects_open_from_wrong_location(self) -> None:
        result = self.validator.validate_mapping(plan(step("open", container="cupboard")), test_world())
        self.assertEqual("ROBOT_NOT_AT_LOCATION", result.issues[0].code)

    def test_rejects_opening_surface(self) -> None:
        result = self.validator.validate_mapping(plan(step("open", container="table")), test_world())
        self.assertEqual("NOT_OPENABLE", result.issues[0].code)

    def test_find_records_observation_without_moving_object(self) -> None:
        result = self.validator.validate_mapping(
            plan(step("find", object="apple")),
            test_world(),
        )
        self.assertTrue(result.valid)
        self.assertEqual(["apple"], result.predicted_final_state["robot"]["observed_objects"])
        final_objects = {
            item["id"]: item for item in result.predicted_final_state["objects"]
        }
        self.assertEqual("table", final_objects["apple"]["location"])

    def test_pick_requires_find(self) -> None:
        result = self.validator.validate_mapping(
            plan(step("pick", object="apple")),
            test_world(),
        )
        self.assertEqual("OBJECT_NOT_OBSERVED", result.issues[0].code)

    def test_find_rejects_unknown_object(self) -> None:
        result = self.validator.validate_mapping(
            plan(step("find", object="banana")),
            test_world(),
        )
        self.assertEqual("UNKNOWN_OBJECT", result.issues[0].code)

    def test_find_rejects_wrong_robot_location(self) -> None:
        result = self.validator.validate_mapping(
            plan(
                step("navigate", location="basket"),
                step("find", object="apple"),
            ),
            test_world(),
        )
        self.assertEqual("ROBOT_NOT_AT_LOCATION", result.issues[0].code)

    def test_find_rejects_object_inside_closed_container(self) -> None:
        world = test_world()
        world.objects["apple"].location = "cupboard"
        world.robot.location = "cupboard"
        result = self.validator.validate_mapping(
            plan(step("find", object="apple")),
            world,
        )
        self.assertEqual("CONTAINER_CLOSED", result.issues[0].code)


if __name__ == "__main__":
    unittest.main()
