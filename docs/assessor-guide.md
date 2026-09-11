# Assessor guide

Download this repository using GitHub's **Code > Download ZIP**, extract it, and open a terminal in the directory containing `pyproject.toml`. The smaller archive produced by `build_submission_archive.py` omits research results and runtime evidence; use the GitHub download for assessment.

## Setup and checks

Install Python 3.10 or newer. Run `./setup.ps1` in PowerShell or `sh setup.sh` on Linux/macOS, then activate `.venv` using the instructions printed by setup. Setup requires internet access to install dependencies.

These checks do not require a Groq API key or Webots:

```text
python -m unittest discover -s tests -v
robot-plan examples/valid_apple_to_basket.json
robot-cnn --run-root cnn_artifacts/kitchen_object_net_v1 verify
python -m robot_planner.evaluation_cli analyse --output evaluation/review-analysis
```

The invalid example `robot-plan examples/invalid_closed_cupboard.json` should be rejected with exit code 1.

For new natural-language commands, add your own Groq API key to `.env` and run `robot-task "Put the apple in the basket"`. For live simulation, install Webots R2025a or the documented Docker runtime and follow the README. The API key, Python environment, and Webots installation are not included.

## Evidence included

- `evaluation/pilot/`: pilot index and trial records.
- `evaluation/results/`: 150 trial records, analysis, metrics, and audit material.
- `evaluation/feedback/`: study materials and a responses CSV. The CSV contains only its header; no completed participant responses are supplied.
- `logs/runs/`: saved execution records.
- `webots/runtime/`: saved plans, results, controller traces, and screenshots referenced by the run records. Historical records contain original absolute paths; find the corresponding files using the path suffix beginning with `webots/runtime/` in this download.
- `cnn_artifacts/kitchen_object_net_v1/`: accepted ONNX model, training configuration, provenance, metrics, plots, six verification images, and compact end-to-end evidence.

## Reproduction limitations

The original 8,000-image training dataset, training checkpoints, and full CNN training/end-to-end logs are not present in the supplied local project or this repository. Normal inference and the portable CNN verifier work without them. Dataset generation and retraining commands are provided in the README, but this download does not enable an independent inspection of all original training images or checkpoints.

The historical `evaluation/frozen-settings.json` does not match five files in the current project: `pyproject.toml`, `config/kitchen.json`, `src/robot_planner/webots_executor.py`, `webots/controllers/robot_task_controller/robot_task_controller.py`, and `webots/worlds/kitchen_llm.wbt`. Consequently, the default fresh evaluation run refuses collection. The historical hashes are preserved as evidence; they have not been rewritten to claim an exact reproduction.

To conduct a separate evaluation of the current code, use a new settings file and output directory (requires a Groq key):

```text
python -m robot_planner.evaluation_cli freeze --output evaluation/current-settings.json
python -m robot_planner.evaluation_cli run --frozen evaluation/current-settings.json --output evaluation/current-results
```

## Local verification on 11 September 2026

Using the GitHub checkout's source with the existing Python 3.10 environment: all 74 unit tests passed, the valid example passed, and CNN verification passed all six fixtures and its integrity checks. A Windows test comparison was corrected to resolve equivalent short and long paths. Dependency installation in a brand-new environment and a live Webots/Groq run were not verified during this review.
