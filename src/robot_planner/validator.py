"""Fail-fast validation of structured plans against a cloned world state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import apply_step
from .errors import ActionValidationError, PlanSchemaError, ValidationIssue
from .models import Plan, WorldState


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    plan_id: str | None
    steps_checked: int
    issues: tuple[ValidationIssue, ...]
    predicted_final_state: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "plan_id": self.plan_id,
            "steps_checked": self.steps_checked,
            "issues": [issue.to_dict() for issue in self.issues],
            "predicted_final_state": self.predicted_final_state,
        }


class PlanValidator:
    """Checks plan structure and sequential feasibility without mutating input state."""

    def validate_mapping(self, raw_plan: object, initial_world: WorldState) -> ValidationResult:
        try:
            plan = Plan.from_mapping(raw_plan)
        except PlanSchemaError as exc:
            return ValidationResult(
                valid=False,
                plan_id=None,
                steps_checked=0,
                issues=(ValidationIssue("INVALID_PLAN_SCHEMA", str(exc), path=exc.path),),
                predicted_final_state=None,
            )
        return self.validate(plan, initial_world)

    def validate(self, plan: Plan, initial_world: WorldState) -> ValidationResult:
        working_state = initial_world.clone()
        for index, step in enumerate(plan.steps):
            try:
                apply_step(working_state, step)
            except ActionValidationError as exc:
                return ValidationResult(
                    valid=False,
                    plan_id=plan.plan_id,
                    steps_checked=index,
                    issues=(ValidationIssue(exc.code, str(exc), step_index=index),),
                    predicted_final_state=None,
                )
        return ValidationResult(
            valid=True,
            plan_id=plan.plan_id,
            steps_checked=len(plan.steps),
            issues=(),
            predicted_final_state=working_state.to_dict(),
        )

