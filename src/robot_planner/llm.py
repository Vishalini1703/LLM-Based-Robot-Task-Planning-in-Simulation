"""Groq-backed natural-language planner with a strict local trust boundary."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from .actions import ACTION_LIBRARY
from .models import WorldState
from .validator import PlanValidator


class PlannerError(RuntimeError):
    """Raised when the remote planner cannot return a usable response."""

    def __init__(
        self,
        message: str,
        attempts: tuple["PlannerAttempt", ...] = (),
    ) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class PlannerAttempt:
    index: int
    latency_ms: float
    status: str
    raw_response: str | None
    validation_issue: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "latency_ms": round(self.latency_ms, 3),
            "status": self.status,
            "raw_response": self.raw_response,
            "validation_issue": self.validation_issue,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class PlannerResponse:
    status: str
    model: str
    latency_ms: float
    raw_response: str
    plan: dict[str, Any] | None
    message: str | None
    usage: dict[str, int]
    attempts: tuple[PlannerAttempt, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.plan is not None

    @property
    def total_latency_ms(self) -> float:
        if self.attempts:
            return sum(attempt.latency_ms for attempt in self.attempts)
        return self.latency_ms

    @property
    def correction_attempted(self) -> bool:
        return len(self.attempts) > 1

    @property
    def correction_succeeded(self) -> bool:
        return self.correction_attempted and self.ready

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 3),
            "raw_response": self.raw_response,
            "plan": self.plan,
            "message": self.message,
            "usage": dict(self.usage),
            "attempt_count": len(self.attempts) or 1,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "correction_attempted": self.correction_attempted,
            "correction_succeeded": self.correction_succeeded,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


Transport = Callable[[str, str, dict[str, Any], float], Mapping[str, Any]]


def _http_transport(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "robot-task-planner/0.2",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", {}).get("message", body)
        except json.JSONDecodeError:
            detail = body
        raise PlannerError(f"Groq API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PlannerError(f"Could not reach Groq API: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise PlannerError("Groq API returned a non-JSON HTTP response.") from exc


class GroqPlanner:
    """Translate one command into a plan proposal or an explicit rejection."""

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 30.0,
        max_attempts: int = 2,
        transport: Transport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key cannot be empty")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._transport = transport or _http_transport

    def plan(self, command: str, world: WorldState) -> PlannerResponse:
        command = command.strip()
        if not command:
            raise PlannerError("Command cannot be empty.")

        requested_plan_id = f"groq-{uuid.uuid4()}"
        payload = self._payload(command, world, requested_plan_id)
        last_error: PlannerError | None = None
        attempts: list[PlannerAttempt] = []

        for attempt in range(self.max_attempts):
            started = time.perf_counter()
            try:
                response = self._transport(self.endpoint, self._api_key, payload, self.timeout)
                latency_ms = (time.perf_counter() - started) * 1000
                planner_response = self._parse_response(response, latency_ms)
                if not planner_response.ready:
                    attempts.append(
                        PlannerAttempt(
                            index=attempt,
                            latency_ms=latency_ms,
                            status="rejected",
                            raw_response=planner_response.raw_response,
                        )
                    )
                    return replace(planner_response, attempts=tuple(attempts))
                validation = PlanValidator().validate_mapping(planner_response.plan, world)
                if validation.valid:
                    attempts.append(
                        PlannerAttempt(
                            index=attempt,
                            latency_ms=latency_ms,
                            status="accepted",
                            raw_response=planner_response.raw_response,
                        )
                    )
                    return replace(planner_response, attempts=tuple(attempts))
                issue = validation.issues[0]
                attempts.append(
                    PlannerAttempt(
                        index=attempt,
                        latency_ms=latency_ms,
                        status="validation_rejected",
                        raw_response=planner_response.raw_response,
                        validation_issue=issue.to_dict(),
                    )
                )
                if attempt + 1 < self.max_attempts:
                    payload["messages"].extend(
                        [
                            {"role": "assistant", "content": planner_response.raw_response},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous plan was rejected by the deterministic validator. "
                                    f"Error {issue.code}: {issue.message} "
                                    "Return a corrected response using the exact response contract. "
                                    "Every step.arguments value must be a JSON object mapping the "
                                    "required argument names to string identifiers."
                                ),
                            },
                        ]
                    )
                    continue
                return replace(planner_response, attempts=tuple(attempts))
            except PlannerError as exc:
                last_error = exc
                attempts.append(
                    PlannerAttempt(
                        index=attempt,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        status="error",
                        raw_response=None,
                        error=str(exc),
                    )
                )
                if attempt + 1 < self.max_attempts:
                    time.sleep(0.25 * (attempt + 1))

        assert last_error is not None
        raise PlannerError(str(last_error), attempts=tuple(attempts)) from last_error

    def _payload(self, command: str, world: WorldState, plan_id: str) -> dict[str, Any]:
        actions = {
            name: {
                "arguments_object": {
                    argument: "string identifier"
                    for argument in sorted(definition.arguments)
                },
                "description": definition.description,
            }
            for name, definition in ACTION_LIBRARY.items()
        }
        system_prompt = (
            "You are a constrained kitchen robot task planner. Return one JSON object only. "
            "Never invent actions, objects, or locations. Use the current state and action definitions. "
            "If the command is ambiguous, unsafe, unsupported, or impossible from the declared world, "
            "return status='rejected', plan=null, and a short message. Otherwise return status='ready', "
            "message=null, and a complete plan. The ready plan must contain exactly plan_id, goal, and "
            "steps. Each step must contain exactly action and arguments. Use navigation before actions "
            "that require the robot at another location. Find an object after arriving at its location "
            "and before picking it. Open closed containers before accessing them."
        )
        user_payload = {
            "command": command,
            "required_plan_id": plan_id,
            "world": world.to_dict(),
            "allowed_actions": actions,
            "response_contract": {
                "status": "ready or rejected",
                "plan": "plan object when ready, otherwise null",
                "message": "null when ready, otherwise rejection reason",
            },
            "exact_ready_example": {
                "status": "ready",
                "plan": {
                    "plan_id": plan_id,
                    "goal": "Put the apple in the basket",
                    "steps": [
                        {"action": "find", "arguments": {"object": "apple"}},
                        {"action": "pick", "arguments": {"object": "apple"}},
                        {
                            "action": "navigate",
                            "arguments": {"location": "basket"},
                        },
                        {
                            "action": "place",
                            "arguments": {
                                "object": "apple",
                                "location": "basket",
                            },
                        },
                    ],
                },
                "message": None,
            },
        }
        return {
            "model": self.model,
            "temperature": 0,
            "max_completion_tokens": 1200,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
            ],
        }

    def _parse_response(
        self, response: Mapping[str, Any], latency_ms: float
    ) -> PlannerResponse:
        try:
            content = response["choices"][0]["message"]["content"]
            response_model = str(response.get("model", self.model))
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerError("Groq response did not contain a chat completion message.") from exc
        if not isinstance(content, str) or not content.strip():
            raise PlannerError("Groq returned an empty planner response.")
        try:
            envelope = json.loads(content)
        except json.JSONDecodeError as exc:
            raise PlannerError("Groq planner response was not valid JSON.") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"status", "plan", "message"}:
            raise PlannerError("Groq planner envelope must contain exactly status, plan, and message.")

        status = envelope["status"]
        plan = envelope["plan"]
        message = envelope["message"]
        if status == "ready":
            if not isinstance(plan, dict) or message is not None:
                raise PlannerError("A ready planner response must contain a plan and message=null.")
        elif status == "rejected":
            if plan is not None or not isinstance(message, str) or not message.strip():
                raise PlannerError("A rejected planner response must contain plan=null and a message.")
            message = message.strip()
        else:
            raise PlannerError("Planner status must be 'ready' or 'rejected'.")

        raw_usage = response.get("usage", {})
        usage = {
            key: int(raw_usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(raw_usage, Mapping)
        }
        return PlannerResponse(
            status=status,
            model=response_model,
            latency_ms=latency_ms,
            raw_response=content,
            plan=plan,
            message=message,
            usage=usage,
        )
