# Structured Plan Contract

The authoritative machine-readable schema is `schemas/plan.schema.json`. Runtime enforcement is also implemented in the Python parser so the core has no external dependency.

## Plan shape

```json
{
  "plan_id": "example-valid-001",
  "goal": "Put the apple in the basket",
  "steps": [
    {
      "action": "pick",
      "arguments": { "object": "apple" }
    },
    {
      "action": "navigate",
      "arguments": { "location": "basket" }
    },
    {
      "action": "place",
      "arguments": { "object": "apple", "location": "basket" }
    }
  ]
}
```

All three top-level fields are required. Unknown fields are rejected. `plan_id` and `goal` must be non-empty strings, and `steps` must contain at least one step. Each step contains exactly `action` and `arguments`.

## Allowed actions

| Action | Required arguments | Meaning |
| --- | --- | --- |
| `navigate` | `location` | Move the robot to a declared location. |
| `find` | `object` | Observe an object at the robot's current location before picking it. |
| `pick` | `object` | Pick up a portable, accessible object at the current location. |
| `place` | `object`, `location` | Place the held object at the current location. |
| `open` | `container` | Open an openable container at the current location. |
| `close` | `container` | Close an openable container at the current location. |


Action names are normalised to lowercase by the Python parser. Identifiers are case-sensitive and must match the world configuration.

## Error codes

| Code | Meaning |
| --- | --- |
| `INVALID_PLAN_SCHEMA` | Required structure, type, or field constraints were violated. |
| `UNKNOWN_ACTION` | The plan invented an action outside the fixed skill library. |
| `INVALID_ARGUMENTS` | An action omitted, added, or emptied an argument. |
| `UNKNOWN_OBJECT` | The named object is absent from the kitchen. |
| `UNKNOWN_LOCATION` | The named location or container is absent. |
| `OBJECT_NOT_PORTABLE` | The plan tried to pick up a fixed object. |
| `HAND_OCCUPIED` | The plan tried to pick while already holding an object. |
| `OBJECT_UNAVAILABLE` | The object has no accessible world position. |
| `ROBOT_NOT_AT_LOCATION` | The action requires navigation to another location first. |
| `OBJECT_NOT_OBSERVED` | The object must be found/observed before it can be picked. |
| `OBJECT_NOT_HELD` | The plan tried to place an object the robot is not holding. |
| `CONTAINER_CLOSED` | The plan tried to access an openable container before opening it. |
| `NOT_OPENABLE` | The plan tried to open or close a surface or permanently open container. |
| `ALREADY_OPEN` | The requested open transition would not change state. |
| `ALREADY_CLOSED` | The requested close transition would not change state. |

## Deliberate constraints

The contract models high-level task planning, not motion planning. It does not contain coordinates, grasp poses, force, speed, collision geometry, or Webots device commands. Those belong to the later simulator adapter and must remain downstream of this validation boundary.

