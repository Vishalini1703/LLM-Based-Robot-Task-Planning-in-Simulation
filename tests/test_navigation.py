import math
import unittest

from robot_planner.navigation import (
    KITCHEN_OBSTACLES,
    ROBOT_CLEARANCE,
    minimum_clearance,
    path_length,
    plan_path,
    segment_is_clear,
)


SERVICE_POINTS = {
    "table": (0.65, -2.55),
    "basket": (0.65, -2.55),
    "shelf": (-1.20, 0.30),
    "cupboard": (1.05, -0.25),
}


class NavigationTests(unittest.TestCase):
    def test_direct_shelf_to_cupboard_route_crosses_furniture(self):
        self.assertFalse(
            segment_is_clear(
                SERVICE_POINTS["shelf"],
                SERVICE_POINTS["cupboard"],
                KITCHEN_OBSTACLES,
                clearance=ROBOT_CLEARANCE,
            )
        )

    def test_planner_routes_around_table_and_chairs(self):
        start = SERVICE_POINTS["shelf"]
        goal = SERVICE_POINTS["cupboard"]
        route = plan_path(start, goal)
        self.assertGreater(path_length(route), math.dist(start, goal))
        self.assertGreaterEqual(minimum_clearance(route), ROBOT_CLEARANCE - 0.01)
        self.assertTrue(
            all(
                segment_is_clear(
                    first,
                    second,
                    KITCHEN_OBSTACLES,
                    clearance=ROBOT_CLEARANCE,
                )
                for first, second in zip(route, route[1:])
            )
        )

    def test_all_service_points_have_routes(self):
        for start_name, start in SERVICE_POINTS.items():
            for goal_name, goal in SERVICE_POINTS.items():
                with self.subTest(start=start_name, goal=goal_name):
                    route = plan_path(start, goal)
                    self.assertEqual(start, route[0])
                    self.assertEqual(goal, route[-1])

    def test_rejects_goal_inside_table(self):
        with self.assertRaisesRegex(ValueError, "goal is inside"):
            plan_path(SERVICE_POINTS["cupboard"], (-0.65, -1.43))
