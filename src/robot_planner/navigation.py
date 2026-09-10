"""Collision-aware 2D path planning for the constrained Webots kitchen."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count
from typing import Iterable


Point2D = tuple[float, float]


@dataclass(frozen=True, slots=True)
class RectangleObstacle:
    name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def expanded(self, clearance: float) -> "RectangleObstacle":
        return RectangleObstacle(
            self.name,
            self.min_x - clearance,
            self.max_x + clearance,
            self.min_y - clearance,
            self.max_y + clearance,
        )

    def contains(self, point: Point2D) -> bool:
        return (
            self.min_x <= point[0] <= self.max_x
            and self.min_y <= point[1] <= self.max_y
        )

    def distance_to(self, point: Point2D) -> float:
        dx = max(self.min_x - point[0], 0.0, point[0] - self.max_x)
        dy = max(self.min_y - point[1], 0.0, point[1] - self.max_y)
        return math.hypot(dx, dy)


KITCHEN_OBSTACLES = (
    RectangleObstacle("dining_table", -1.15, -0.15, -2.33, -0.53),
    RectangleObstacle("chair_north", -0.89, -0.39, -0.75, -0.25),
    RectangleObstacle("chair_south", -0.89, -0.39, -2.58, -2.08),
    RectangleObstacle("chair_east_north", -0.43, 0.07, -1.30, -0.80),
    RectangleObstacle("chair_east_south", -0.43, 0.07, -2.12, -1.62),
    RectangleObstacle("chair_west_north", -1.34, -0.84, -1.30, -0.80),
    RectangleObstacle("chair_west_south", -1.37, -0.87, -2.12, -1.62),
    RectangleObstacle("kitchen_counter", 1.45, 1.65, -1.22, 0.98),
    RectangleObstacle("north_counter", -0.35, 1.65, 0.98, 1.18),
    RectangleObstacle("fridge", 1.50, 2.10, -1.95, -1.20),
)
# Bounds describe valid robot-centre positions, not the room's wall surfaces.
# The west limit leaves room for TIAGo's circular base beside the wall.
KITCHEN_BOUNDS = (-1.45, 1.08, -2.60, 0.70)
ROBOT_CLEARANCE = 0.38


def segment_is_clear(
    start: Point2D,
    end: Point2D,
    obstacles: Iterable[RectangleObstacle],
    *,
    clearance: float,
    sample_spacing: float = 0.025,
) -> bool:
    expanded = tuple(obstacle.expanded(clearance) for obstacle in obstacles)
    distance = math.dist(start, end)
    samples = max(1, math.ceil(distance / sample_spacing))
    for index in range(samples + 1):
        ratio = index / samples
        point = (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )
        if any(obstacle.contains(point) for obstacle in expanded):
            return False
    return True


def _point_in_bounds(point: Point2D, bounds: tuple[float, float, float, float]) -> bool:
    min_x, max_x, min_y, max_y = bounds
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y


def _path_length(points: Iterable[Point2D]) -> float:
    sequence = tuple(points)
    return sum(math.dist(start, end) for start, end in zip(sequence, sequence[1:]))


def _smooth_path(
    points: list[Point2D],
    obstacles: tuple[RectangleObstacle, ...],
    clearance: float,
) -> list[Point2D]:
    if len(points) <= 2:
        return points
    smoothed = [points[0]]
    current = 0
    while current < len(points) - 1:
        candidate = len(points) - 1
        while candidate > current + 1:
            if segment_is_clear(
                points[current],
                points[candidate],
                obstacles,
                clearance=clearance,
            ):
                break
            candidate -= 1
        smoothed.append(points[candidate])
        current = candidate
    return smoothed


def plan_path(
    start: Point2D,
    goal: Point2D,
    *,
    obstacles: tuple[RectangleObstacle, ...] = KITCHEN_OBSTACLES,
    clearance: float = ROBOT_CLEARANCE,
    bounds: tuple[float, float, float, float] = KITCHEN_BOUNDS,
    resolution: float = 0.10,
) -> list[Point2D]:
    """Plan and smooth an eight-connected A* route around expanded obstacles."""
    if not _point_in_bounds(start, bounds) or not _point_in_bounds(goal, bounds):
        raise ValueError("Navigation start and goal must be inside kitchen bounds.")
    if not segment_is_clear(start, start, obstacles, clearance=clearance):
        raise ValueError(f"Navigation start is inside an obstacle: {start}.")
    if not segment_is_clear(goal, goal, obstacles, clearance=clearance):
        raise ValueError(f"Navigation goal is inside an obstacle: {goal}.")
    if segment_is_clear(start, goal, obstacles, clearance=clearance):
        return [start, goal]

    min_x, max_x, min_y, max_y = bounds
    width = round((max_x - min_x) / resolution)
    height = round((max_y - min_y) / resolution)

    def to_node(point: Point2D) -> tuple[int, int]:
        return (
            round((point[0] - min_x) / resolution),
            round((point[1] - min_y) / resolution),
        )

    def to_point(node: tuple[int, int]) -> Point2D:
        return (
            round(min_x + node[0] * resolution, 6),
            round(min_y + node[1] * resolution, 6),
        )

    start_node = to_node(start)
    goal_node = to_node(goal)
    expanded = tuple(obstacle.expanded(clearance) for obstacle in obstacles)

    def node_is_clear(node: tuple[int, int]) -> bool:
        if not (0 <= node[0] <= width and 0 <= node[1] <= height):
            return False
        point = to_point(node)
        return not any(obstacle.contains(point) for obstacle in expanded)

    frontier: list[tuple[float, int, tuple[int, int]]] = []
    sequence = count()
    heapq.heappush(frontier, (0.0, next(sequence), start_node))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    cost_so_far = {start_node: 0.0}
    moves = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal_node:
            break
        for dx, dy in moves:
            neighbour = (current[0] + dx, current[1] + dy)
            if not node_is_clear(neighbour):
                continue
            if dx and dy:
                if not node_is_clear((current[0] + dx, current[1])):
                    continue
                if not node_is_clear((current[0], current[1] + dy)):
                    continue
            step_cost = math.sqrt(2.0) if dx and dy else 1.0
            new_cost = cost_so_far[current] + step_cost
            if neighbour in cost_so_far and new_cost >= cost_so_far[neighbour]:
                continue
            cost_so_far[neighbour] = new_cost
            priority = new_cost + math.dist(neighbour, goal_node)
            heapq.heappush(frontier, (priority, next(sequence), neighbour))
            came_from[neighbour] = current

    if goal_node not in came_from:
        raise RuntimeError(f"No collision-free route from {start} to {goal}.")

    nodes = []
    current: tuple[int, int] | None = goal_node
    while current is not None:
        nodes.append(current)
        current = came_from[current]
    nodes.reverse()
    points = [start, *(to_point(node) for node in nodes[1:-1]), goal]
    smoothed = _smooth_path(points, obstacles, clearance)
    if any(
        not segment_is_clear(a, b, obstacles, clearance=clearance)
        for a, b in zip(smoothed, smoothed[1:])
    ):
        raise RuntimeError("Path smoothing created an unsafe navigation segment.")
    return smoothed


def path_length(path: Iterable[Point2D]) -> float:
    return _path_length(path)


def minimum_clearance(
    path: Iterable[Point2D],
    obstacles: tuple[RectangleObstacle, ...] = KITCHEN_OBSTACLES,
    *,
    sample_spacing: float = 0.025,
) -> float:
    points = tuple(path)
    minimum = math.inf
    for start, end in zip(points, points[1:]):
        distance = math.dist(start, end)
        samples = max(1, math.ceil(distance / sample_spacing))
        for index in range(samples + 1):
            ratio = index / samples
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
            minimum = min(
                minimum,
                *(obstacle.distance_to(point) for obstacle in obstacles),
            )
    return minimum
