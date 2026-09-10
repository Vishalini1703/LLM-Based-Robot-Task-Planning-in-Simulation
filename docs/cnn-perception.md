# Custom CNN perception extension

## Control boundary

The natural-language objective and six-action plan contract are unchanged.
Groq-hosted Llama 3.3 produces only a high-level JSON plan. The deterministic
validator checks that complete plan before Webots starts. `KitchenObjectNet`
is consulted only inside `find(object)` and cannot add a plan step, move the
robot, operate a motor, change the world model, or override a failed
precondition. `pick(object)` remains dependent on a successful visual `find`.

Normal CNN execution disables Webots recognition and asserts its sampling
period is zero. The `metadata` perception backend is an explicit diagnostic
option; it is never a fallback. Missing models, corrupt models, inference
errors, inconsistent frames, and absent requested classes fail closed with
`OBJECT_NOT_VISUALLY_DETECTED`.

## Dataset

The deterministic controller mode generates 8,000 640 by 480 RGB frames in
the adapted Webots R2025a kitchen. The immutable protocol contains 5,600 train,
1,200 validation, and 1,200 test frames. Exactly 1,600 frames are negatives.
The five classes are `mug`, `apple`, `orange`, `can`, and `cereal_box`.

Webots recognition supplies ground-truth boxes only during generation. Each
saved positive frame is rejected and regenerated unless every selected object
has a complete bounding box with a four-pixel image margin. Seeds, difficulty,
split, image path, dimensions, class IDs, and boxes are stored in project-owned
JSON Lines. `dataset_manifest.json` retains the annotation/world hashes and all
8,000 image hashes. The audit rejects missing data, invalid/out-of-image boxes,
duplicates, class under-coverage, incorrect split sizes, an incorrect negative
count, or seed leakage.

The generated content is a simulation output derived from the included Webots
world and Cyberbotics assets. No internet photographs or external object
detection dataset is used. Webots assets remain subject to the Cyberbotics
Webots asset licence; project Python code and learned weights are submission
artefacts for academic use.

## Original architecture

`KitchenObjectNet` is implemented with basic PyTorch layers in this repository.
It has five convolutional stages producing 40 by 40, 20 by 20, and 10 by 10
features. A project-defined top-down pyramid fuses those features into a 40 by
40 detection grid. Four heads predict objectness, five class logits, box width
and height, and centre-cell offsets. Decoding, class-wise non-maximum
suppression, target assignment, balanced focal objectness loss, cross-entropy
classification loss, CIoU box loss, and offset loss are project code.

The model has 1,927,562 trainable parameters, below the six-million parameter
budget. Backbone/fusion convolutions use seeded Kaiming initialization;
prediction heads use seeded small-normal initialization so the initial detector
is uninformative and numerically stable. Batch-normalizer scales start at one
and biases at zero. There is no model-download function,
checkpoint input, pretrained option, TorchVision detector, YOLO dependency,
COCO weight, or ImageNet weight. The training provenance files hash the initial
random state, selected checkpoint, annotations, and ONNX model.

## Training and fair evaluation

Training uses only the train split for gradient updates and augmentation. The
validation split controls early stopping, checkpoint selection, and the
confidence threshold. The test loader is constructed only after both the model
and threshold have been frozen. The first mixed-precision attempt is retained
in the logs; its non-finite gradients led to a fail-fast change to full FP32.
Full precision is used on CUDA and CPU, with fatal checks for non-finite loss or
gradients.

The artifact folder contains epoch-level loss components, validation metrics,
the untouched test report, per-class/difficulty metrics, threshold sweep,
confusion matrix, prediction examples, CPU ONNX latency, PyTorch/ONNX parity,
and model/checkpoint hashes. The supplemental 180-episode protocol selects 30
distinct held-out Webots scenes for each class and 30 negatives, balanced ten
each across easy, medium, and hard conditions.

### Accepted measured result

Training completed for 25 epochs and selected epoch 13 using validation mAP.
The frozen validation-selected confidence threshold is 0.35. On the untouched
1,200-image test split, the accepted model obtained mAP@0.50 of 0.9918,
mAP@0.75 of 0.9390, macro precision of 0.9979, macro recall of 0.9927, and zero
false detections over 242 negative images. All five classes exceeded the 0.85
precision/recall acceptance floor. ONNX output differed from PyTorch by at most
`4.76837158203125e-06`; the measured 100-sample CPU median was approximately
8.07 ms on the validation computer.

The independently frozen 180-scene protocol passed 173/180: mug 28/30, apple
28/30, orange 29/30, can 28/30, cereal box 30/30, and negatives 30/30. This
protocol is supplemental evidence and was not used to select weights or the
confidence threshold.

## Runtime inference

Webots BGRA bytes are converted to RGB, resized to 320 by 320, mapped from
`[0, 1]` to `[-1, 1]`, and passed to the frozen ONNX model on CPU. At each
bounded head scan pose, the requested class must exceed the validation-selected
threshold in two of three frames with image-box IoU of at least 0.10. Evidence
contains the class, confidence, normalized and pixel box, model SHA-256,
threshold, inference time, scan pose, and an annotated RGB image.

This detector is intended for the constrained simulated kitchen only. It is
not evidence of real-world generalization, collision safety, or suitability for
a physical robot.

## Runtime and packaging verification

The final end-to-end protocol starts a fresh Webots process and freshly loaded
world for every episode, matching the normal one-command client lifecycle and
preventing physics state from leaking between repetitions. It completed 25/25
episodes—five for each class—with zero non-zero process exits. Every episode
contained a verified CNN-backed `find`, all observations used the accepted
ONNX SHA-256
`02d0d5496d183ee006147354cf0e76159dd090e4b3e5672fd04a1a5d27251a96`,
recognition metadata was used zero times, and all 35 navigation paths met the
required obstacle clearance (minimum observed clearance 0.40 m).

`robot-cnn verify` checks the model/provenance hashes, rejected imported-model
paths, CPU-only ONNX loading, an empty black frame, five positive fixture images
and one negative fixture, the 25-run summary, plots, and retained logs. The
submission archive includes the ONNX model, class map, threshold/evaluation,
portable manifest, compact end-to-end summary, six fixtures, plots, source,
tests, Webots project, Docker recipe, and technical documentation. It excludes
the large generated dataset, checkpoints, full runtime traces, API key, and
machine-specific paths; those full materials remain under the local
`cnn_artifacts/kitchen_object_net_v1/` folder.
