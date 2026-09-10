"""Typed errors returned by plan parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class WorldConfigurationError(ValueError):
    """Raised when a kitchen-world configuration is internally inconsistent."""


class PlanSchemaError(ValueError):
    """Raised when a plan does not satisfy the machine-readable contract."""

    def __init__(self, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.path = path


class ActionValidationError(ValueError):
    """Raised when an action cannot be applied to the current world state."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A stable, serialisable description of a rejected plan."""

    code: str
    message: str
    step_index: int | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.step_index is not None:
            result["step_index"] = self.step_index
        if self.path is not None:
            result["path"] = self.path
        return result

