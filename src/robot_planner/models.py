"""Immutable plan contracts and mutable, cloneable kitchen state models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .errors import PlanSchemaError


class LocationKind(str, Enum):
    SURFACE = "surface"
    CONTAINER = "container"


@dataclass(slots=True)
class KitchenLocation:
    id: str
    kind: LocationKind
    openable: bool
    is_open: bool

    @property
    def is_accessible(self) -> bool:
        return not self.openable or self.is_open

    def clone(self) -> "KitchenLocation":
        return KitchenLocation(self.id, self.kind, self.openable, self.is_open)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "openable": self.openable,
            "is_open": self.is_open,
        }


@dataclass(slots=True)
class KitchenObject:
    id: str
    location: str | None
    portable: bool

    def clone(self) -> "KitchenObject":
        return KitchenObject(self.id, self.location, self.portable)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "location": self.location, "portable": self.portable}


@dataclass(slots=True)
class RobotState:
    location: str
    held_object: str | None = None
    observed_objects: set[str] | None = None

    def __post_init__(self) -> None:
        if self.observed_objects is None:
            self.observed_objects = set()

    def clone(self) -> "RobotState":
        return RobotState(self.location, self.held_object, set(self.observed_objects or ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "held_object": self.held_object,
            "observed_objects": sorted(self.observed_objects or ()),
        }


@dataclass(slots=True)
class WorldState:
    locations: dict[str, KitchenLocation]
    objects: dict[str, KitchenObject]
    robot: RobotState

    def clone(self) -> "WorldState":
        return WorldState(
            locations={key: value.clone() for key, value in self.locations.items()},
            objects={key: value.clone() for key, value in self.objects.items()},
            robot=self.robot.clone(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "locations": [self.locations[key].to_dict() for key in sorted(self.locations)],
            "objects": [self.objects[key].to_dict() for key in sorted(self.objects)],
            "robot": self.robot.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PlanStep:
    action: str
    arguments: dict[str, str]

    @classmethod
    def from_mapping(cls, value: object, path: str) -> "PlanStep":
        if not isinstance(value, Mapping):
            raise PlanSchemaError("Each step must be an object.", path)
        unexpected = set(value) - {"action", "arguments"}
        missing = {"action", "arguments"} - set(value)
        if missing:
            raise PlanSchemaError(f"Missing required field(s): {', '.join(sorted(missing))}.", path)
        if unexpected:
            raise PlanSchemaError(f"Unexpected field(s): {', '.join(sorted(unexpected))}.", path)

        action = value["action"]
        arguments = value["arguments"]
        if not isinstance(action, str) or not action.strip():
            raise PlanSchemaError("Action must be a non-empty string.", f"{path}.action")
        if not isinstance(arguments, Mapping):
            raise PlanSchemaError("Arguments must be an object.", f"{path}.arguments")
        if not all(isinstance(key, str) and isinstance(item, str) for key, item in arguments.items()):
            raise PlanSchemaError(
                "Argument names and values must be strings.", f"{path}.arguments"
            )
        return cls(action=action.strip().lower(), arguments=dict(arguments))

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "arguments": dict(self.arguments)}


@dataclass(frozen=True, slots=True)
class Plan:
    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...]

    @classmethod
    def from_mapping(cls, value: object) -> "Plan":
        if not isinstance(value, Mapping):
            raise PlanSchemaError("Plan must be a JSON object.")
        expected = {"plan_id", "goal", "steps"}
        missing = expected - set(value)
        unexpected = set(value) - expected
        if missing:
            raise PlanSchemaError(f"Missing required field(s): {', '.join(sorted(missing))}.")
        if unexpected:
            raise PlanSchemaError(f"Unexpected field(s): {', '.join(sorted(unexpected))}.")

        plan_id = value["plan_id"]
        goal = value["goal"]
        raw_steps = value["steps"]
        if not isinstance(plan_id, str) or not plan_id.strip():
            raise PlanSchemaError("plan_id must be a non-empty string.", "$.plan_id")
        if not isinstance(goal, str) or not goal.strip():
            raise PlanSchemaError("goal must be a non-empty string.", "$.goal")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanSchemaError("steps must be a non-empty array.", "$.steps")

        steps = tuple(
            PlanStep.from_mapping(step, f"$.steps[{index}]")
            for index, step in enumerate(raw_steps)
        )
        return cls(plan_id=plan_id.strip(), goal=goal.strip(), steps=steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
        }
