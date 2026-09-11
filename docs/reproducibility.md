# Reproducibility Guide

## Clean setup

Install Python 3.10 or newer, create a virtual environment, install with `python -m pip install -e .`, and place a Groq key in an untracked `.env` file as `GROQ_AI_KEY` or `GROQ_API_KEY`. No client path or GPU model is hard-coded.

For headless simulation, install Docker and pull `cyberbotics/webots:R2025a-ubuntu22.04`. Native Webots is discovered from the CLI option, `WEBOTS_BIN`, `WEBOTS_HOME`, or `PATH`.

## Verification sequence

```powershell
python -m unittest discover -s tests -v
robot-plan examples/valid_apple_to_basket.json
robot-task "Put the apple in the basket"
robot-task "Put the apple in the basket" --backend webots --webots-runtime docker
robot-cnn --run-root cnn_artifacts/kitchen_object_net_v1 verify
```

The code-only submission intentionally excludes raw research results. To
collect a fresh evaluation and generate its metrics:

```powershell
python -m robot_planner.evaluation_cli run
python -m robot_planner.evaluation_cli analyse
python tools/extended_analysis.py
```

The frozen-input verifier refuses collection if a frozen code, world, schema, or catalogue hash differs.

## CNN reproduction and evidence

Install `.[cnn-training]` only when regenerating the dataset/model. Run
`robot-cnn generate`, `audit`, `train`, and `protocol` in the order documented
in the README. Training begins from seeded random weights and accepts no
checkpoint or pretrained-weight argument. Normal inference needs only NumPy,
Pillow, ONNX Runtime, and the included `KitchenObjectNet.onnx`.

The accepted ONNX SHA-256 is
`02d0d5496d183ee006147354cf0e76159dd090e4b3e5672fd04a1a5d27251a96`.
`robot-cnn verify` checks this hash, provenance, CPU execution provider, five
positive fixtures, one negative fixture, the frozen 180-scene report, eight
plots, and the compact 25-run Webots summary. The full local artifact directory
retains all 8,000 images, checkpoints, failed attempts, logs, screenshots, and
plots.

## Recorded environment

- Evaluation interpreter: Python 3.12.0 (project minimum remains Python 3.10).
- Docker client: 29.6.2.
- Webots image: `cyberbotics/webots:R2025a-ubuntu22.04`.
- Image digest: `sha256:f0023e30daf38b172e4e6ad24ed345909bcd9551df34d63d824e121a7cebf099`.
- Exact experimental hashes and collection time: `evaluation/frozen-settings.json`.

## Portable archive

Build and integrity-check a reproduction ZIP with:

```powershell
python tools/build_repro_archive.py
```

The builder excludes `.env`, caches, local runtime logs, and the output archive itself. It refuses to package text containing a Groq credential pattern or an absolute Windows machine path.

Build the smaller ready-to-submit code and code-documentation archive with:

```powershell
python tools/build_submission_archive.py
```

This archive includes the accepted ONNX model, portable evidence, plots and six
CPU verification fixtures, while excluding the large dataset, checkpoints,
full runtime logs, thesis/proposal files, and `.env`.
