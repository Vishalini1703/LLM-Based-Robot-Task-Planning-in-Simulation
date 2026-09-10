"""Loading and integrity checks for constrained kitchen configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .errors import WorldConfigurationError
from .models import KitchenLocation, KitchenObject, LocationKind, RobotState, WorldState


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorldConfigurationError(f"{label} must be an object.")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unexpected = set(value) - expected
    if missing:
        raise WorldConfigurationError(f"{label} is missing: {', '.join(sorted(missing))}.")
    if unexpected:
        raise WorldConfigurationError(f"{label} has unexpected fields: {', '.join(sorted(unexpected))}.")


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldConfigurationError(f"{label} must be a non-empty string.")
    return value.strip()


def world_from_mapping(raw: object) -> WorldState:
    data = _require_mapping(raw, "World")
    _require_exact_keys(data, {"locations", "objects", "robot"}, "World")
    if not isinstance(data["locations"], list) or not data["locations"]:
        raise WorldConfigurationError("locations must be a non-empty array.")
    if not isinstance(data["objects"], list):
        raise WorldConfigurationError("objects must be an array.")

    locations: dict[str, KitchenLocation] = {}
    for index, item in enumerate(data["locations"]):
        location_data = _require_mapping(item, f"locations[{index}]")
        _require_exact_keys(
            location_data, {"id", "kind", "openable", "is_open"}, f"locations[{index}]"
        )
        location_id = _require_identifier(location_data["id"], f"locations[{index}].id")
        if location_id in locations:
            raise WorldConfigurationError(f"Duplicate location id: {location_id}.")
        try:
            kind = LocationKind(location_data["kind"])
        except (ValueError, TypeError) as exc:
            raise WorldConfigurationError(
                f"locations[{index}].kind must be 'surface' or 'container'."
            ) from exc
        openable = location_data["openable"]
        is_open = location_data["is_open"]
        if not isinstance(openable, bool) or not isinstance(is_open, bool):
            raise WorldConfigurationError(
                f"locations[{index}].openable and is_open must be booleans."
            )
        if kind is LocationKind.SURFACE and openable:
            raise WorldConfigurationError(f"Surface '{location_id}' cannot be openable.")
        if not openable and not is_open:
            raise WorldConfigurationError(
                f"Non-openable location '{location_id}' must remain accessible (is_open=true)."
            )
        locations[location_id] = KitchenLocation(location_id, kind, openable, is_open)

    objects: dict[str, KitchenObject] = {}
    for index, item in enumerate(data["objects"]):
        object_data = _require_mapping(item, f"objects[{index}]")
        _require_exact_keys(object_data, {"id", "location", "portable"}, f"objects[{index}]")
        object_id = _require_identifier(object_data["id"], f"objects[{index}].id")
        if object_id in objects:
            raise WorldConfigurationError(f"Duplicate object id: {object_id}.")
        location = object_data["location"]
        if location is not None:
            location = _require_identifier(location, f"objects[{index}].location")
            if location not in locations:
                raise WorldConfigurationError(
                    f"Object '{object_id}' refers to unknown location '{location}'."
                )
        portable = object_data["portable"]
        if not isinstance(portable, bool):
            raise WorldConfigurationError(f"objects[{index}].portable must be a boolean.")
        objects[object_id] = KitchenObject(object_id, location, portable)

    robot_data = _require_mapping(data["robot"], "robot")
    required_robot_fields = {"location", "held_object"}
    missing_robot_fields = required_robot_fields - set(robot_data)
    unexpected_robot_fields = set(robot_data) - required_robot_fields - {"observed_objects"}
    if missing_robot_fields:
        raise WorldConfigurationError(
            f"robot is missing: {', '.join(sorted(missing_robot_fields))}."
        )
    if unexpected_robot_fields:
        raise WorldConfigurationError(
            f"robot has unexpected fields: {', '.join(sorted(unexpected_robot_fields))}."
        )
    robot_location = _require_identifier(robot_data["location"], "robot.location")
    if robot_location not in locations:
        raise WorldConfigurationError(f"Robot refers to unknown location '{robot_location}'.")
    held_object = robot_data["held_object"]
    if held_object is not None:
        held_object = _require_identifier(held_object, "robot.held_object")
        if held_object not in objects:
            raise WorldConfigurationError(f"Robot holds unknown object '{held_object}'.")
        if objects[held_object].location is not None:
            raise WorldConfigurationError(
                f"Held object '{held_object}' must have location=null."
            )
    unlocated = [item.id for item in objects.values() if item.location is None]
    if held_object is None and unlocated:
        raise WorldConfigurationError(
            f"Unlocated object(s) are not held by the robot: {', '.join(sorted(unlocated))}."
        )
    if held_object is not None and unlocated != [held_object]:
        raise WorldConfigurationError("Exactly the held object must have location=null.")

    raw_observed = robot_data.get("observed_objects", [])
    if not isinstance(raw_observed, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_observed
    ):
        raise WorldConfigurationError(
            "robot.observed_objects must be an array of non-empty object identifiers."
        )
    observed_objects = {item.strip() for item in raw_observed}
    unknown_observed = observed_objects - set(objects)
    if unknown_observed:
        raise WorldConfigurationError(
            "Robot observations refer to unknown object(s): "
            + ", ".join(sorted(unknown_observed))
            + "."
        )

    return WorldState(
        locations,
        objects,
        RobotState(robot_location, held_object, observed_objects),
    )


def load_world(path: str | Path) -> WorldState:
    world_path = Path(path)
    try:
        with world_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except json.JSONDecodeError as exc:
        raise WorldConfigurationError(
            f"Invalid JSON in {world_path}: line {exc.lineno}, column {exc.colno}."
        ) from exc
    return world_from_mapping(raw)
