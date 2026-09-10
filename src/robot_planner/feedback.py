"""Validation and aggregate analysis for anonymous participant feedback."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


RATING_FIELDS = (
    "clarity",
    "usability",
    "understandability",
    "usefulness",
    "rejection_confidence",
)
TEXT_FIELDS = (
    "clearest_feature",
    "most_confusing_feature",
    "suggested_improvement",
)
REQUIRED_FIELDS = ("participant_id", *RATING_FIELDS, *TEXT_FIELDS)
PARTICIPANT_ID = re.compile(r"^P\d{3}$")


def load_feedback(path: str | Path) -> list[dict[str, Any]]:
    """Load and strictly validate the anonymous questionnaire CSV."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError(
                "Feedback columns must be exactly: " + ", ".join(REQUIRED_FIELDS)
            )
        rows = []
        seen_ids: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            participant_id = (raw["participant_id"] or "").strip()
            if not PARTICIPANT_ID.fullmatch(participant_id):
                raise ValueError(
                    f"Line {line_number}: participant_id must match P001 format."
                )
            if participant_id in seen_ids:
                raise ValueError(
                    f"Line {line_number}: duplicate participant_id {participant_id}."
                )
            seen_ids.add(participant_id)
            row: dict[str, Any] = {"participant_id": participant_id}
            for field in RATING_FIELDS:
                try:
                    rating = int((raw[field] or "").strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Line {line_number}: {field} must be an integer from 1 to 5."
                    ) from exc
                if rating not in range(1, 6):
                    raise ValueError(
                        f"Line {line_number}: {field} must be from 1 to 5."
                    )
                row[field] = rating
            for field in TEXT_FIELDS:
                row[field] = (raw[field] or "").strip()
            rows.append(row)
    return rows


def summarise_feedback(
    rows: list[dict[str, Any]],
    *,
    invited_count: int,
) -> dict[str, Any]:
    """Return aggregate ratings without exposing participant-level records."""
    if invited_count < len(rows):
        raise ValueError("invited_count cannot be lower than completed responses.")
    if invited_count < 1:
        raise ValueError("invited_count must be positive.")
    ratings = {}
    for field in RATING_FIELDS:
        values = [int(row[field]) for row in rows]
        ratings[field] = {
            "count": len(values),
            "mean": round(statistics.fmean(values), 3) if values else None,
            "median": statistics.median(values) if values else None,
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "distribution": {
                str(score): sum(value == score for value in values)
                for score in range(1, 6)
            },
        }
    comments = {
        field: [row[field] for row in rows if row[field]]
        for field in TEXT_FIELDS
    }
    return {
        "invited_count": invited_count,
        "completed_count": len(rows),
        "completion_rate_percent": round(100 * len(rows) / invited_count, 3),
        "ratings": ratings,
        "anonymous_comments": comments,
        "privacy": {
            "participant_ids_included": False,
            "participant_level_ratings_included": False,
        },
    }


def analyse_feedback_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    invited_count: int,
) -> dict[str, Any]:
    summary = summarise_feedback(
        load_feedback(input_path),
        invited_count=invited_count,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
