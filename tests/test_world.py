from __future__ import annotations

import unittest

from robot_planner.errors import WorldConfigurationError
from robot_planner.world import world_from_mapping


def base_world() -> dict:
    return {
        "locations": [
            {"id": "table", "kind": "surface", "openable": False, "is_open": True},
            {"id": "cupboard", "kind": "container", "openable": True, "is_open": False},
        ],
        "objects": [{"id": "apple", "location": "table", "portable": True}],
        "robot": {"location": "table", "held_object": None},
    }


class WorldLoadingTests(unittest.TestCase):
    def test_loads_consistent_world(self) -> None:
        world = world_from_mapping(base_world())
        self.assertEqual("table", world.robot.location)
        self.assertEqual("table", world.objects["apple"].location)
        self.assertFalse(world.locations["cupboard"].is_open)

    def test_clone_is_independent(self) -> None:
        world = world_from_mapping(base_world())
        cloned = world.clone()
        cloned.robot.location = "cupboard"
        cloned.locations["cupboard"].is_open = True
        self.assertEqual("table", world.robot.location)
        self.assertFalse(world.locations["cupboard"].is_open)

    def test_rejects_duplicate_location(self) -> None:
        raw = base_world()
        raw["locations"].append(
            {"id": "table", "kind": "surface", "openable": False, "is_open": True}
        )
        with self.assertRaisesRegex(WorldConfigurationError, "Duplicate location"):
            world_from_mapping(raw)

    def test_rejects_unknown_object_location(self) -> None:
        raw = base_world()
        raw["objects"][0]["location"] = "moon"
        with self.assertRaisesRegex(WorldConfigurationError, "unknown location"):
            world_from_mapping(raw)

    def test_rejects_unheld_unlocated_object(self) -> None:
        raw = base_world()
        raw["objects"][0]["location"] = None
        with self.assertRaisesRegex(WorldConfigurationError, "not held"):
            world_from_mapping(raw)


if __name__ == "__main__":
    unittest.main()

