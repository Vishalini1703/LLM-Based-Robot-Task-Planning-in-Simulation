"""Validated in-memory execution with complete step-level audit records."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .actions import apply_step
from .models import Plan, WorldState
from .validator import PlanValidator, ValidationResult


class ExecutionRejected(RuntimeError):
    def __init__(self, validation: ValidationResult) -> None:
        super().__init__("Plan execution was blocked by validation.")
        self.validation = validation


class ExecutionObservationError(RuntimeError):
    def __init__(
        self,
        step_index: int,
        expected_state: dict[str, Any],
        observed_state: dict[str, Any],
    ) -> None:
        super().__init__(
            f"Observed state did not match the expected state after step {step_index}."
        )
        self.code = "OBSERVED_STATE_MISMATCH"
        self.step_index = step_index
        self.expected_state = expected_state
        self.observed_state = observed_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "failed",
            "code": self.code,
            "message": str(self),
            "step_index": self.step_index,
            "expected_state": self.expected_state,
            "observed_state": self.observed_state,
        }


@dataclass(frozen=True, slots=True)
class StepExecution:
    index: int
    action: str
    arguments: dict[str, str]
    duration_ms: float
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    expected_state: dict[str, Any]
    observed_state: dict[str, Any]
    outcome_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "arguments": dict(self.arguments),
            "duration_ms": round(self.duration_ms, 3),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "expected_state": self.expected_state,
            "observed_state": self.observed_state,
            "outcome_verified": self.outcome_verified,
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    run_id: str
    plan_id: str
    started_at: str
    duration_ms: float
    steps: tuple[StepExecution, ...]
    final_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "steps": [step.to_dict() for step in self.steps],
            "final_state": self.final_state,
        }


class InMemoryExecutor:
    """Execute high-level actions after independently rechecking the full plan."""

    def __init__(
        self,
        validator: PlanValidator | None = None,
        observer: Callable[[WorldState], dict[str, Any]] | None = None,
    ) -> None:
        self.validator = validator or PlanValidator()
        self.observer = observer or (lambda state: state.to_dict())

    def execute(self, plan: Plan, initial_world: WorldState) -> ExecutionResult:
        validation = self.validator.validate(plan, initial_world)
        if not validation.valid:
            raise ExecutionRejected(validation)

        state = initial_world.clone()
        run_started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        records: list[StepExecution] = []
        for index, step in enumerate(plan.steps):
            before = state.to_dict()
            step_started = time.perf_counter()
            apply_step(state, step)
            duration_ms = (time.perf_counter() - step_started) * 1000
            expected_state = state.to_dict()
            observed_state = self.observer(state)
            outcome_verified = observed_state == expected_state
            if not outcome_verified:
                raise ExecutionObservationError(
                    index,
                    expected_state,
                    observed_state,
                )
            records.append(
                StepExecution(
                    index=index,
                    action=step.action,
                    arguments=dict(step.arguments),
                    duration_ms=duration_ms,
                    state_before=before,
                    state_after=observed_state,
                    expected_state=expected_state,
                    observed_state=observed_state,
                    outcome_verified=outcome_verified,
                )
            )
        return ExecutionResult(
            run_id=str(uuid.uuid4()),
            plan_id=plan.plan_id,
            started_at=started_at,
            duration_ms=(time.perf_counter() - run_started) * 1000,
            steps=tuple(records),
            final_state=state.to_dict(),
        )
