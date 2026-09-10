"""End-to-end command planning, validation, execution, and run persistence."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .executor import ExecutionResult, InMemoryExecutor
from .llm import GroqPlanner, PlannerError, PlannerResponse
from .models import Plan, WorldState
from .validator import PlanValidator, ValidationResult


class ExecutionBackend(Protocol):
    def execute(self, plan: Plan, initial_world: WorldState) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class PipelineFailure:
    stage: str
    code: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    run_id: str
    command: str
    created_at: str
    status: str
    initial_state: dict[str, Any]
    duration_ms: float
    planner: PlannerResponse | None
    validation: ValidationResult | None
    execution: Any | None
    failure: PipelineFailure | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "command": self.command,
            "created_at": self.created_at,
            "status": self.status,
            "initial_state": self.initial_state,
            "duration_ms": round(self.duration_ms, 3),
            "planner": self.planner.to_dict() if self.planner else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "execution": self.execution.to_dict() if self.execution else None,
            "failure": self.failure.to_dict() if self.failure else None,
        }


class TaskPlanningPipeline:
    def __init__(
        self,
        planner: GroqPlanner,
        validator: PlanValidator | None = None,
        executor: ExecutionBackend | None = None,
    ) -> None:
        self.planner = planner
        self.validator = validator or PlanValidator()
        self.executor = executor or InMemoryExecutor(self.validator)

    def run(self, command: str, world: WorldState) -> PipelineResult:
        run_started = time.perf_counter()
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        initial_state = world.to_dict()
        try:
            planner_response = self.planner.plan(command, world)
        except PlannerError as exc:
            failure = PipelineFailure(
                stage="planner",
                code="PLANNER_ERROR",
                message=str(exc),
                details={
                    "attempts": [attempt.to_dict() for attempt in exc.attempts],
                },
            )
            return PipelineResult(
                run_id=run_id,
                command=command,
                created_at=created_at,
                status="planner_failed",
                initial_state=initial_state,
                duration_ms=(time.perf_counter() - run_started) * 1000,
                planner=None,
                validation=None,
                execution=None,
                failure=failure,
            )
        if not planner_response.ready:
            return PipelineResult(
                run_id=run_id,
                command=command,
                created_at=created_at,
                status="planner_rejected",
                initial_state=initial_state,
                duration_ms=(time.perf_counter() - run_started) * 1000,
                planner=planner_response,
                validation=None,
                execution=None,
            )

        validation = self.validator.validate_mapping(planner_response.plan, world)
        if not validation.valid:
            return PipelineResult(
                run_id=run_id,
                command=command,
                created_at=created_at,
                status="validation_rejected",
                initial_state=initial_state,
                duration_ms=(time.perf_counter() - run_started) * 1000,
                planner=planner_response,
                validation=validation,
                execution=None,
            )

        plan = Plan.from_mapping(planner_response.plan)
        try:
            execution = self.executor.execute(plan, world)
        except Exception as exc:
            details = exc.to_dict() if hasattr(exc, "to_dict") else {}
            failure = PipelineFailure(
                stage="execution",
                code=str(getattr(exc, "code", type(exc).__name__)),
                message=str(exc),
                details=details,
            )
            return PipelineResult(
                run_id=run_id,
                command=command,
                created_at=created_at,
                status="execution_failed",
                initial_state=initial_state,
                duration_ms=(time.perf_counter() - run_started) * 1000,
                planner=planner_response,
                validation=validation,
                execution=None,
                failure=failure,
            )
        return PipelineResult(
            run_id=run_id,
            command=command,
            created_at=created_at,
            status="executed",
            initial_state=initial_state,
            duration_ms=(time.perf_counter() - run_started) * 1000,
            planner=planner_response,
            validation=validation,
            execution=execution,
        )


def save_pipeline_result(result: PipelineResult, directory: str | Path) -> Path:
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{result.run_id}.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return output_path
