from __future__ import annotations

import unittest
from pathlib import Path

from robot_planner.cnn_e2e import _steps_for
from robot_planner.models import Plan
from robot_planner.validator import PlanValidator
from robot_planner.world import load_world


class CnnEndToEndPlanTests(unittest.TestCase):
    def test_all_five_fixed_manipulation_plans_pass_the_real_validator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        world = load_world(root / "config" / "kitchen.json")
        for object_id in ("mug", "apple", "orange", "can", "cereal_box"):
            with self.subTest(object_id=object_id):
                plan = Plan.from_mapping(
                    {
                        "plan_id": f"test-{object_id}",
                        "goal": f"Manipulate {object_id}",
                        "steps": _steps_for(object_id),
                    }
                )
                validation = PlanValidator().validate(plan, world)
                self.assertTrue(validation.valid, validation.to_dict())


if __name__ == "__main__":
    unittest.main()
