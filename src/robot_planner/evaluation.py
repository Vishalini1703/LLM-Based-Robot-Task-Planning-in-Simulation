"""Frozen experiment execution and reproducible metric analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .llm import GroqPlanner
from .pipeline import TaskPlanningPipeline
from .webots_executor import WebotsExecutor
from .world import load_world


HALLUCINATION_CODES = {"UNKNOWN_ACTION", "UNKNOWN_OBJECT", "UNKNOWN_LOCATION"}


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    category: str
    command: str
    expected_executable: bool
    expected_statuses: tuple[str, ...]
    expected_issue_codes: tuple[str, ...]
    expected_state: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EvaluationCase":
        required = {
            "id",
            "category",
            "command",
            "expected_executable",
            "expected_statuses",
            "expected_issue_codes",
            "expected_state",
        }
        if set(value) != required:
            raise ValueError(
                "Evaluation case fields must be exactly: "
                + ", ".join(sorted(required))
            )
        if not all(
            isinstance(value[field], str) and value[field].strip()
            for field in ("id", "category", "command")
        ):
            raise ValueError("Evaluation case id, category, and command must be non-empty.")
        if not isinstance(value["expected_executable"], bool):
            raise ValueError("expected_executable must be boolean.")
        if not isinstance(value["expected_statuses"], list) or not all(
            isinstance(item, str) for item in value["expected_statuses"]
        ):
            raise ValueError("expected_statuses must be an array of strings.")
        if not isinstance(value["expected_issue_codes"], list) or not all(
            isinstance(item, str) for item in value["expected_issue_codes"]
        ):
            raise ValueError("expected_issue_codes must be an array of strings.")
        if not isinstance(value["expected_state"], dict):
            raise ValueError("expected_state must be an object.")
        return cls(
            id=value["id"].strip(),
            category=value["category"].strip(),
            command=value["command"].strip(),
            expected_executable=value["expected_executable"],
            expected_statuses=tuple(value["expected_statuses"]),
            expected_issue_codes=tuple(value["expected_issue_codes"]),
            expected_state=value["expected_state"],
        )


def load_catalog(path: str | Path) -> tuple[EvaluationCase, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"version", "cases"}:
        raise ValueError("Catalogue must contain exactly version and cases.")
    if not isinstance(raw["cases"], list):
        raise ValueError("Catalogue cases must be an array.")
    cases = tuple(EvaluationCase.from_mapping(item) for item in raw["cases"])
    identifiers = [case.id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evaluation case identifiers must be unique.")
    if len(cases) != 30:
        raise ValueError(f"Frozen evaluation catalogue must contain 30 cases, got {len(cases)}.")
    category_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1
    required_counts = {
        "valid_single": 6,
        "valid_multi": 8,
        "invalid_precondition": 6,
        "unsupported": 5,
        "ambiguous": 5,
    }
    if category_counts != required_counts:
        raise ValueError(
            f"Catalogue category counts are {category_counts}, expected {required_counts}."
        )
    return cases


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_settings(
    project_root: str | Path,
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    frozen_files = protocol.get("frozen_files")
    if not isinstance(frozen_files, list) or not frozen_files:
        raise ValueError("Protocol frozen_files must be a non-empty array.")
    hashes = {}
    for relative in frozen_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen file does not exist: {path}")
        hashes[str(relative).replace("\\", "/")] = sha256_file(path)
    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": protocol["version"],
        "model": protocol["model"],
        "temperature": protocol["temperature"],
        "max_completion_tokens": protocol["max_completion_tokens"],
        "max_attempts": protocol["max_attempts"],
        "timeout_seconds": protocol["timeout_seconds"],
        "repetitions": protocol["repetitions"],
        "backend": protocol["backend"],
        "hashes": hashes,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def verify_frozen_settings(
    project_root: str | Path,
    frozen_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in frozen["hashes"].items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(
                {"file": relative, "expected": expected, "actual": actual}
            )
    if mismatches:
        raise RuntimeError(
            "Frozen evaluation inputs changed: " + json.dumps(mismatches)
        )
    return frozen


def run_offline_tests(project_root: str | Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=Path(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Evaluation preflight tests failed:\n"
            + completed.stdout[-4000:]
            + completed.stderr[-4000:]
        )
    combined = completed.stdout + completed.stderr
    return {
        "returncode": completed.returncode,
        "output_tail": combined[-4000:],
    }


def _objects_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in state.get("objects", [])}


def _locations_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in state.get("locations", [])}


def state_matches(actual: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if actual is None:
        return False
    objects = _objects_by_id(actual)
    for object_id, location in expected.get("object_locations", {}).items():
        if objects.get(object_id, {}).get("location") != location:
            return False
    locations = _locations_by_id(actual)
    for location_id, is_open in expected.get("location_open", {}).items():
        if locations.get(location_id, {}).get("is_open") is not is_open:
            return False
    robot = actual.get("robot", {})
    expected_robot = expected.get("robot", {})
    for key, value in expected_robot.items():
        if robot.get(key) != value:
            return False
    return True


def _final_state(result: dict[str, Any]) -> dict[str, Any] | None:
    execution = result.get("execution")
    if not isinstance(execution, dict):
        return None
    if isinstance(execution.get("final_state"), dict):
        return execution["final_state"]
    nested = execution.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("final_state"), dict):
        return nested["final_state"]
    return None


def _issue_code(result: dict[str, Any]) -> str | None:
    validation = result.get("validation")
    if isinstance(validation, dict) and validation.get("issues"):
        return validation["issues"][0].get("code")
    failure = result.get("failure")
    if isinstance(failure, dict):
        return failure.get("code")
    return None


def assess_trial(case: EvaluationCase, result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status")
    issue_code = _issue_code(result)
    expected_status = status in case.expected_statuses
    issue_acceptable = (
        not case.expected_issue_codes
        or status == "planner_rejected"
        or issue_code in case.expected_issue_codes
    )
    final_state = _final_state(result)
    task_success = (
        case.expected_executable
        and status == "executed"
        and state_matches(final_state, case.expected_state)
    )
    correct_rejection = (
        not case.expected_executable
        and status in {"planner_rejected", "validation_rejected"}
        and expected_status
        and issue_acceptable
    )
    return {
        "observed_status": status,
        "observed_issue_code": issue_code,
        "expected_status": expected_status,
        "expected_issue_acceptable": issue_acceptable,
        "task_success": task_success,
        "correct_rejection": correct_rejection,
        "classification_correct": task_success
        if case.expected_executable
        else correct_rejection,
    }


def _executor_for_backend(
    backend: str,
    project_root: Path,
    webots_runtime: str,
    webots_bin: str | Path | None,
):
    if backend == "memory":
        return None
    if backend == "webots":
        return WebotsExecutor(
            webots_bin,
            project_root,
            runtime=webots_runtime,
        )
    raise ValueError("Evaluation backend must be memory or webots.")


def run_pilot(
    *,
    project_root: str | Path,
    catalog_path: str | Path,
    world_path: str | Path,
    protocol_path: str | Path,
    output_directory: str | Path,
    api_key: str,
) -> dict[str, Any]:
    """Run one non-final trial from each frozen command category."""
    root = Path(project_root).resolve()
    cases = load_catalog(catalog_path)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    selected: list[EvaluationCase] = []
    selected_categories: set[str] = set()
    for case in cases:
        if case.category not in selected_categories:
            selected.append(case)
            selected_categories.add(case.category)

    preflight = run_offline_tests(root)
    output = Path(output_directory)
    trials_directory = output / "trials"
    trials_directory.mkdir(parents=True, exist_ok=True)
    index = []
    for case in selected:
        world = load_world(world_path)
        planner = GroqPlanner(
            api_key,
            model=protocol["model"],
            timeout=float(protocol["timeout_seconds"]),
            max_attempts=int(protocol["max_attempts"]),
        )
        payload = TaskPlanningPipeline(planner).run(case.command, world).to_dict()
        assessment = assess_trial(case, payload)
        trial_id = f"pilot-{case.id}"
        trial_path = trials_directory / f"{trial_id}.json"
        record = {
            "pilot": True,
            "included_in_final_dataset": False,
            "trial_id": trial_id,
            "case": {
                "id": case.id,
                "category": case.category,
                "command": case.command,
                "expected_executable": case.expected_executable,
                "expected_statuses": list(case.expected_statuses),
                "expected_issue_codes": list(case.expected_issue_codes),
                "expected_state": case.expected_state,
            },
            "result": payload,
            "assessment": assessment,
        }
        trial_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        index.append(
            {
                "trial_id": trial_id,
                "case_id": case.id,
                "category": case.category,
                "status": payload["status"],
                "classification_correct": assessment["classification_correct"],
                "path": str(trial_path),
            }
        )
    result = {
        "pilot": True,
        "included_in_final_dataset": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "trial_count": len(index),
        "all_classifications_correct": all(
            item["classification_correct"] for item in index
        ),
        "trials": index,
    }
    (output / "pilot-index.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_evaluation(
    *,
    project_root: str | Path,
    catalog_path: str | Path,
    world_path: str | Path,
    frozen_path: str | Path,
    output_directory: str | Path,
    api_key: str,
    repetitions: int,
    backend: str,
    webots_runtime: str = "docker",
    webots_bin: str | Path | None = None,
    resume: bool = True,
    run_preflight_tests: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    cases = load_catalog(catalog_path)
    frozen = verify_frozen_settings(root, frozen_path)
    if repetitions != frozen["repetitions"]:
        raise ValueError(
            f"Repetitions must match frozen value {frozen['repetitions']}."
        )
    if backend != frozen["backend"]:
        raise ValueError(f"Backend must match frozen value {frozen['backend']}.")
    preflight = (
        run_offline_tests(root)
        if run_preflight_tests
        else {"returncode": None, "output_tail": "skipped"}
    )
    output = Path(output_directory)
    trials_directory = output / "trials"
    trials_directory.mkdir(parents=True, exist_ok=True)
    trial_index = []
    started_at = datetime.now(timezone.utc).isoformat()

    for repetition in range(1, repetitions + 1):
        for case in cases:
            trial_id = f"{case.id}-r{repetition:02d}"
            trial_path = trials_directory / f"{trial_id}.json"
            if resume and trial_path.is_file():
                record = json.loads(trial_path.read_text(encoding="utf-8"))
                trial_index.append(
                    {
                        "trial_id": trial_id,
                        "case_id": case.id,
                        "repetition": repetition,
                        "path": str(trial_path),
                        "status": record["result"]["status"],
                        "classification_correct": record["assessment"][
                            "classification_correct"
                        ],
                    }
                )
                continue

            world = load_world(world_path)
            executor = _executor_for_backend(
                backend, root, webots_runtime, webots_bin
            )
            planner = GroqPlanner(
                api_key,
                model=frozen["model"],
                timeout=float(frozen["timeout_seconds"]),
                max_attempts=int(frozen["max_attempts"]),
            )
            result = TaskPlanningPipeline(planner, executor=executor).run(
                case.command,
                world,
            )
            payload = result.to_dict()
            assessment = assess_trial(case, payload)
            record = {
                "trial_id": trial_id,
                "case": {
                    "id": case.id,
                    "category": case.category,
                    "command": case.command,
                    "expected_executable": case.expected_executable,
                    "expected_statuses": list(case.expected_statuses),
                    "expected_issue_codes": list(case.expected_issue_codes),
                    "expected_state": case.expected_state,
                },
                "repetition": repetition,
                "backend": backend,
                "frozen_settings_sha256": sha256_file(frozen_path),
                "result": payload,
                "assessment": assessment,
            }
            trial_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            trial_index.append(
                {
                    "trial_id": trial_id,
                    "case_id": case.id,
                    "repetition": repetition,
                    "path": str(trial_path),
                    "status": payload["status"],
                    "classification_correct": assessment["classification_correct"],
                }
            )
            (output / "trial-index.json").write_text(
                json.dumps(trial_index, indent=2),
                encoding="utf-8",
            )
            time.sleep(0.05)

    index_record = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "expected_trials": len(cases) * repetitions,
        "actual_trials": len(trial_index),
        "preflight": preflight,
        "trials": trial_index,
    }
    (output / "trial-index.json").write_text(
        json.dumps(index_record, indent=2),
        encoding="utf-8",
    )
    return index_record


def _percent(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 3) if denominator else None


def _latency_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = sorted(float(value) for value in values)
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "p95": None,
        }
    p95_index = max(0, math.ceil(0.95 * len(numbers)) - 1)
    return {
        "count": len(numbers),
        "mean": round(statistics.fmean(numbers), 3),
        "median": round(statistics.median(numbers), 3),
        "standard_deviation": round(statistics.pstdev(numbers), 3),
        "minimum": round(numbers[0], 3),
        "maximum": round(numbers[-1], 3),
        "p95": round(numbers[p95_index], 3),
    }


def _failure_category(record: dict[str, Any]) -> str:
    result = record["result"]
    status = result["status"]
    issue = record["assessment"].get("observed_issue_code")
    if issue == "INVALID_PLAN_SCHEMA":
        return "schema_error"
    if issue == "UNKNOWN_ACTION":
        return "unsupported_action"
    if issue in {"UNKNOWN_OBJECT", "UNKNOWN_LOCATION"}:
        return "missing_entity"
    if issue in {
        "OBJECT_NOT_PORTABLE",
        "HAND_OCCUPIED",
        "OBJECT_UNAVAILABLE",
        "ROBOT_NOT_AT_LOCATION",
        "OBJECT_NOT_HELD",
        "OBJECT_NOT_OBSERVED",
        "CONTAINER_CLOSED",
        "NOT_OPENABLE",
        "ALREADY_OPEN",
        "ALREADY_CLOSED",
    }:
        return "precondition_failure"
    if status == "execution_failed":
        return "execution_mismatch"
    if status == "planner_failed":
        return "api_or_planner_failure"
    if (
        not record["case"]["expected_executable"]
        and status == "executed"
    ):
        return "incorrect_acceptance"
    if record["case"]["expected_executable"] and status != "executed":
        return "incorrect_rejection"
    return "none"


def analyse_evaluation(
    trials_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    trial_paths = sorted(Path(trials_directory).glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in trial_paths]
    if not records:
        raise ValueError("No evaluation trial JSON files were found.")

    completed_api_calls = len(records)
    structured = sum(record["result"].get("planner") is not None for record in records)
    ready = [
        record
        for record in records
        if (record["result"].get("planner") or {}).get("status") == "ready"
    ]
    valid_ready = sum(
        bool((record["result"].get("validation") or {}).get("valid"))
        for record in ready
    )
    executable = [
        record for record in records if record["case"]["expected_executable"]
    ]
    task_successes = sum(
        bool(record["assessment"]["task_success"]) for record in executable
    )
    rejection_expected = [
        record for record in records if not record["case"]["expected_executable"]
    ]
    correct_rejections = sum(
        bool(record["assessment"]["correct_rejection"])
        for record in rejection_expected
    )
    generated_steps = sum(
        len(
            (
                (record["result"].get("planner") or {}).get("plan") or {}
            ).get("steps", [])
        )
        for record in records
    )
    invalid_steps = sum(
        bool(
            (record["result"].get("validation") or {}).get("issues")
            and (record["result"]["validation"]["issues"][0].get("step_index") is not None)
        )
        for record in records
    )
    hallucinations = sum(
        record["assessment"].get("observed_issue_code") in HALLUCINATION_CODES
        for record in records
    )
    validated_started = sum(
        bool((record["result"].get("validation") or {}).get("valid"))
        for record in records
    )
    execution_errors = sum(
        record["result"]["status"] == "execution_failed" for record in records
    )
    corrections = sum(
        bool((record["result"].get("planner") or {}).get("correction_attempted"))
        for record in records
    )
    correction_successes = sum(
        bool((record["result"].get("planner") or {}).get("correction_succeeded"))
        for record in records
    )
    model_latencies = [
        record["result"]["planner"]["total_latency_ms"]
        for record in records
        if record["result"].get("planner")
    ]
    end_to_end_latencies = [
        record["result"]["duration_ms"] for record in records
    ]

    category_breakdown: dict[str, dict[str, Any]] = {}
    for record in records:
        category = record["case"]["category"]
        bucket = category_breakdown.setdefault(
            category,
            {"trials": 0, "classification_correct": 0, "task_success": 0},
        )
        bucket["trials"] += 1
        bucket["classification_correct"] += int(
            record["assessment"]["classification_correct"]
        )
        bucket["task_success"] += int(record["assessment"]["task_success"])
    for bucket in category_breakdown.values():
        bucket["classification_accuracy_percent"] = _percent(
            bucket["classification_correct"], bucket["trials"]
        )

    failure_taxonomy: dict[str, int] = {}
    for record in records:
        category = _failure_category(record)
        failure_taxonomy[category] = failure_taxonomy.get(category, 0) + 1

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trial_count": len(records),
        "structured_response": {
            "numerator": structured,
            "denominator": completed_api_calls,
            "percent": _percent(structured, completed_api_calls),
        },
        "plan_validity": {
            "numerator": valid_ready,
            "denominator": len(ready),
            "percent": _percent(valid_ready, len(ready)),
        },
        "task_success": {
            "numerator": task_successes,
            "denominator": len(executable),
            "percent": _percent(task_successes, len(executable)),
        },
        "correct_rejection": {
            "numerator": correct_rejections,
            "denominator": len(rejection_expected),
            "percent": _percent(correct_rejections, len(rejection_expected)),
        },
        "invalid_step": {
            "numerator": invalid_steps,
            "denominator": generated_steps,
            "percent": _percent(invalid_steps, generated_steps),
        },
        "hallucination": {
            "numerator": hallucinations,
            "denominator": len(records),
            "percent": _percent(hallucinations, len(records)),
        },
        "execution_error": {
            "numerator": execution_errors,
            "denominator": validated_started,
            "percent": _percent(execution_errors, validated_started),
        },
        "correction_attempt": {
            "numerator": corrections,
            "denominator": len(records),
            "percent": _percent(corrections, len(records)),
        },
        "correction_success": {
            "numerator": correction_successes,
            "denominator": corrections,
            "percent": _percent(correction_successes, corrections),
        },
        "model_latency_ms": _latency_summary(model_latencies),
        "end_to_end_latency_ms": _latency_summary(end_to_end_latencies),
        "category_breakdown": category_breakdown,
        "failure_taxonomy": failure_taxonomy,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    _write_trials_csv(records, output / "trials.csv")
    (output / "summary.md").write_text(_summary_markdown(metrics), encoding="utf-8")
    _write_audit_sample(records, output / "audit-sample.json")
    return metrics


def _write_trials_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "trial_id",
                "case_id",
                "category",
                "repetition",
                "status",
                "issue_code",
                "classification_correct",
                "task_success",
                "correct_rejection",
                "attempt_count",
                "model_latency_ms",
                "end_to_end_latency_ms",
            ],
        )
        writer.writeheader()
        for record in records:
            planner = record["result"].get("planner") or {}
            writer.writerow(
                {
                    "trial_id": record["trial_id"],
                    "case_id": record["case"]["id"],
                    "category": record["case"]["category"],
                    "repetition": record["repetition"],
                    "status": record["result"]["status"],
                    "issue_code": record["assessment"].get("observed_issue_code"),
                    "classification_correct": record["assessment"][
                        "classification_correct"
                    ],
                    "task_success": record["assessment"]["task_success"],
                    "correct_rejection": record["assessment"]["correct_rejection"],
                    "attempt_count": planner.get("attempt_count"),
                    "model_latency_ms": planner.get("total_latency_ms"),
                    "end_to_end_latency_ms": record["result"]["duration_ms"],
                }
            )


def _summary_markdown(metrics: dict[str, Any]) -> str:
    rows = []
    for key in (
        "structured_response",
        "plan_validity",
        "task_success",
        "correct_rejection",
        "invalid_step",
        "hallucination",
        "execution_error",
        "correction_attempt",
        "correction_success",
    ):
        metric = metrics[key]
        value = "N/A" if metric["percent"] is None else f"{metric['percent']:.3f}%"
        rows.append(
            f"| {key.replace('_', ' ').title()} | "
            f"{metric['numerator']} / {metric['denominator']} | {value} |"
        )
    return (
        "# Experimental Evaluation Summary\n\n"
        f"Trials analysed: {metrics['trial_count']}\n\n"
        "| Metric | Result | Rate |\n"
        "| --- | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\n"
        "The figures describe performance only in the frozen constrained "
        "simulation protocol and are not evidence of real-world robot safety.\n"
    )


def _write_audit_sample(records: list[dict[str, Any]], path: Path) -> None:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_category.setdefault(record["case"]["category"], []).append(record)
    target = max(1, math.ceil(len(records) * 0.20))
    selected = []
    while len(selected) < target:
        added = False
        for category in sorted(by_category):
            candidates = by_category[category]
            index = len([item for item in selected if item["category"] == category])
            if index < len(candidates) and len(selected) < target:
                record = candidates[index]
                selected.append(
                    {
                        "trial_id": record["trial_id"],
                        "category": category,
                        "source_file": f"trials/{record['trial_id']}.json",
                        "audit_status": "pending_manual_review",
                        "reviewer_notes": "",
                    }
                )
                added = True
        if not added:
            break
    path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
