# Backend Architecture — Evaluated Prototype

## Control boundary

```text
Natural-language command
   |
   v
Groq structured planner ----> explicit rejection / retained correction attempts
   |
   v
Strict plan parser ----------> schema rejection
   |
   v
Clone initial kitchen state
   |
   v
Apply preconditions and state transitions in sequence
   |                         |
   | valid                   | invalid
   v                         v
Predicted final state     stable error result
   |
   v
Executor revalidation
   |
   v
Memory or Webots execution
   |
   v
Expected-versus-observed evidence per step
```

The LLM proposes only. It cannot add skills, command Webots, bypass validation,
or silently change a plan after validation.

## Modules

- `models.py` defines strict locations, objects, robot state, and plans.
- `world.py` loads configuration and rejects inconsistent initial states.
- `actions.py` owns the fixed skill library, preconditions, and transitions.
- `validator.py` dry-runs a plan on a clone and returns stable issue codes.
- `llm.py` calls Groq, preserves raw attempts, and supports one bounded correction.
- `pipeline.py` joins planning, validation, execution, failures, and timings.
- `executor.py` revalidates and compares expected with independently observed state.
- `navigation.py` plans and smooths clearance-aware A* routes around the kitchen's fixed furniture.
- `cnn_model.py` defines the project-designed randomly initialised detector, losses, decoding, and non-maximum suppression.
- `cnn_data.py` audits the deterministic 8,000-frame Webots dataset and its provenance manifest.
- `perception.py` performs CPU-only ONNX inference, box decoding, and temporal confirmation from raw RGB frames.
- `webots_executor.py` launches discovered native Webots or Docker/Xvfb.
- `evaluation.py` freezes, runs, resumes, indexes, and analyses the experiment.

## Action semantics

`navigate(location)` requires a declared destination and changes the robot's
logical location. In Webots, the adapter expands each fixed obstacle by the
TIAGo base clearance, computes an eight-connected A* route, rejects unsafe
segments, and logs the route length and minimum clearance. `find(object)`
requires the robot to be at the object's
accessible location and records an observation without moving the object.
Webots implements this with raw TIAGo camera frames, the frozen
`KitchenObjectNet` ONNX model, bounded scan poses, and two-of-three temporal
confirmation. Webots recognition is disabled and asserted inactive during
normal CNN execution; metadata is available only as an explicit diagnostic or
dataset/evaluation ground-truth backend.

`pick(object)` requires the object to have been found, to be portable and
co-located, and the robot hand to be empty. `place(object, location)` requires
the exact object to be held, the robot to be at the destination, and any
openable destination to be open. `open(container)` and `close(container)`
require an openable, co-located container and a real state change.

## Observed execution

Every Webots action records physical evidence: robot position, CNN class,
confidence and image box, model hash, annotated image, carried-object position,
placement containment or surface distance, and cupboard-door rotation. The
controller stops at the first mismatch. The executor also requires all step
outcomes to be verified and the observed final logical state to equal the
validator prediction.

The simulation uses high-level supervised path following and object attachment.
Its navigation avoids mapped static furniture, but it is not dynamic obstacle
avoidance, collision-aware grasp synthesis, or proof of physical-robot safety.
