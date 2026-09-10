"""Deterministic action preconditions and state transitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .errors import ActionValidationError
from .models import LocationKind, PlanStep, WorldState


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    arguments: frozenset[str]
    description: str


ACTION_LIBRARY: dict[str, ActionDefinition] = {
    "navigate": ActionDefinition(frozenset({"location"}), "Move the robot to a known location."),
    "find": ActionDefinition(
        frozenset({"object"}),
        "Identify an accessible object at the robot's current location.",
    ),
    "pick": ActionDefinition(frozenset({"object"}), "Pick up an accessible portable object."),
    "place": ActionDefinition(
        frozenset({"object", "location"}), "Place the held object at an accessible location."
    ),
    "open": ActionDefinition(frozenset({"container"}), "Open an openable container."),
    "close": ActionDefinition(frozenset({"container"}), "Close an openable container."),
}


def _check_arguments(step: PlanStep) -> None:
    definition = ACTION_LIBRARY.get(step.action)
    if definition is None:
        raise ActionValidationError(
            "UNKNOWN_ACTION",
            f"Action '{step.action}' is not in the allowed skill library.",
        )
    actual = set(step.arguments)
    missing = definition.arguments - actual
    unexpected = actual - definition.arguments
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected {', '.join(sorted(unexpected))}")
        raise ActionValidationError(
            "INVALID_ARGUMENTS",
            f"Action '{step.action}' has invalid arguments: {'; '.join(details)}.",
        )
    if any(not value.strip() for value in step.arguments.values()):
        raise ActionValidationError("INVALID_ARGUMENTS", "Action arguments cannot be empty.")


def _location(world: WorldState, location_id: str):
    location = world.locations.get(location_id)
    if location is None:
        raise ActionValidationError(
            "UNKNOWN_LOCATION", f"Location '{location_id}' does not exist in the kitchen."
        )
    return location


def _object(world: WorldState, object_id: str):
    item = world.objects.get(object_id)
    if item is None:
        raise ActionValidationError(
            "UNKNOWN_OBJECT", f"Object '{object_id}' does not exist in the kitchen."
        )
    return item


def _require_robot_at(world: WorldState, location_id: str) -> None:
    if world.robot.location != location_id:
        raise ActionValidationError(
            "ROBOT_NOT_AT_LOCATION",
            f"Robot is at '{world.robot.location}', not '{location_id}'.",
        )


def _require_accessible(world: WorldState, location_id: str) -> None:
    location = _location(world, location_id)
    if not location.is_accessible:
        raise ActionValidationError(
            "CONTAINER_CLOSED", f"Container '{location_id}' must be opened first."
        )


def _navigate(world: WorldState, arguments: dict[str, str]) -> None:
    target = arguments["location"]
    _location(world, target)
    world.robot.location = target
    world.robot.observed_objects = set()


def _find(world: WorldState, arguments: dict[str, str]) -> None:
    object_id = arguments["object"]
    item = _object(world, object_id)
    if item.location is None:
        raise ActionValidationError(
            "OBJECT_UNAVAILABLE", f"Object '{object_id}' has no world location."
        )
    _require_robot_at(world, item.location)
    _require_accessible(world, item.location)
    assert world.robot.observed_objects is not None
    world.robot.observed_objects.add(object_id)


def _pick(world: WorldState, arguments: dict[str, str]) -> None:
    object_id = arguments["object"]
    item = _object(world, object_id)
    if not item.portable:
        raise ActionValidationError("OBJECT_NOT_PORTABLE", f"Object '{object_id}' is not portable.")
    if world.robot.held_object is not None:
        raise ActionValidationError(
            "HAND_OCCUPIED", f"Robot is already holding '{world.robot.held_object}'."
        )
    if item.location is None:
        raise ActionValidationError("OBJECT_UNAVAILABLE", f"Object '{object_id}' has no world location.")
    _require_robot_at(world, item.location)
    _require_accessible(world, item.location)
    if object_id not in (world.robot.observed_objects or set()):
        raise ActionValidationError(
            "OBJECT_NOT_OBSERVED",
            f"Object '{object_id}' must be found before it can be picked.",
        )
    item.location = None
    world.robot.held_object = object_id
    world.robot.observed_objects.discard(object_id)


def _place(world: WorldState, arguments: dict[str, str]) -> None:
    object_id = arguments["object"]
    destination = arguments["location"]
    item = _object(world, object_id)
    _location(world, destination)
    if world.robot.held_object != object_id:
        held = world.robot.held_object or "nothing"
        raise ActionValidationError(
            "OBJECT_NOT_HELD", f"Robot is holding '{held}', not '{object_id}'."
        )
    _require_robot_at(world, destination)
    _require_accessible(world, destination)
    item.location = destination
    world.robot.held_object = None
    world.robot.observed_objects = {object_id}


def _open(world: WorldState, arguments: dict[str, str]) -> None:
    container_id = arguments["container"]
    container = _location(world, container_id)
    _require_robot_at(world, container_id)
    if container.kind is not LocationKind.CONTAINER or not container.openable:
        raise ActionValidationError(
            "NOT_OPENABLE", f"Location '{container_id}' is not an openable container."
        )
    if container.is_open:
        raise ActionValidationError("ALREADY_OPEN", f"Container '{container_id}' is already open.")
    container.is_open = True


def _close(world: WorldState, arguments: dict[str, str]) -> None:
    container_id = arguments["container"]
    container = _location(world, container_id)
    _require_robot_at(world, container_id)
    if container.kind is not LocationKind.CONTAINER or not container.openable:
        raise ActionValidationError(
            "NOT_OPENABLE", f"Location '{container_id}' is not an openable container."
        )
    if not container.is_open:
        raise ActionValidationError("ALREADY_CLOSED", f"Container '{container_id}' is already closed.")
    container.is_open = False
    world.robot.observed_objects.difference_update(
        item.id for item in world.objects.values() if item.location == container_id
    )


_HANDLERS: dict[str, Callable[[WorldState, dict[str, str]], None]] = {
    "navigate": _navigate,
    "find": _find,
    "pick": _pick,
    "place": _place,
    "open": _open,
    "close": _close,
}


def apply_step(world: WorldState, step: PlanStep) -> None:
    """Validate and apply one action atomically to the supplied state."""

    _check_arguments(step)
    _HANDLERS[step.action](world, step.arguments)
