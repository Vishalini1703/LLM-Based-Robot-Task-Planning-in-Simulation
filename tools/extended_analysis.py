"""Create reconciled breakdowns and an SVG figure from frozen trial records."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def wilson_interval(successes: int, total: int) -> dict[str, float | None]:
    if total == 0:
        return {"lower_percent": None, "upper_percent": None}
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return {
        "lower_percent": round(100 * (centre - margin), 3),
        "upper_percent": round(100 * (centre + margin), 3),
    }


def _rate(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    successes = sum(bool(record["assessment"][field]) for record in records)
    total = len(records)
    return {
        "successes": successes,
        "trials": total,
        "percent": round(100 * successes / total, 3) if total else None,
        "wilson_95_percent": wilson_interval(successes, total),
    }


def analyse(trials_directory: Path) -> dict[str, Any]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(trials_directory.glob("*.json"))
    ]
    trial_ids = [record["trial_id"] for record in records]
    schedule = {
        (record["case"]["id"], int(record["repetition"])) for record in records
    }
    expected_schedule = {
        (record["case"]["id"], repetition)
        for record in records
        for repetition in range(1, 6)
    }
    executable = [record for record in records if record["case"]["expected_executable"]]
    rejected = [record for record in records if not record["case"]["expected_executable"]]

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "category": defaultdict(list),
        "attempt_count": defaultdict(list),
        "generated_plan_length": defaultdict(list),
        "backend": defaultdict(list),
    }
    for record in records:
        planner = record["result"].get("planner") or {}
        plan = planner.get("plan") or {}
        length = len(plan.get("steps", []))
        grouped["category"][record["case"]["category"]].append(record)
        grouped["attempt_count"][str(planner.get("attempt_count", 0))].append(record)
        grouped["generated_plan_length"][str(length)].append(record)
        grouped["backend"][record["backend"]].append(record)

    breakdowns = {}
    for dimension, buckets in grouped.items():
        breakdowns[dimension] = {
            name: _rate(bucket, "classification_correct")
            for name, bucket in sorted(buckets.items())
        }
    return {
        "reconciliation": {
            "trial_files": len(records),
            "unique_trial_ids": len(set(trial_ids)),
            "duplicate_trial_ids": sorted(
                trial_id for trial_id in set(trial_ids) if trial_ids.count(trial_id) > 1
            ),
            "expected_schedule_entries": len(expected_schedule),
            "observed_schedule_entries": len(schedule),
            "missing_schedule_entries": sorted(
                f"{case_id}-r{repetition:02d}"
                for case_id, repetition in expected_schedule - schedule
            ),
        },
        "task_success": _rate(executable, "task_success"),
        "correct_rejection": _rate(rejected, "correct_rejection"),
        "breakdowns": breakdowns,
    }


def category_svg(analysis: dict[str, Any]) -> str:
    categories = analysis["breakdowns"]["category"]
    labels = list(categories)
    width, height = 920, 430
    left, top, chart_height = 230, 45, 300
    bar_height = 42
    colours = {
        "valid_single": "#2563eb",
        "valid_multi": "#0f766e",
        "invalid_precondition": "#7c3aed",
        "unsupported": "#c2410c",
        "ambiguous": "#b91c1c",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-family="Arial" font-size="20" font-weight="700">Classification accuracy by frozen command category</text>',
    ]
    for index, label in enumerate(labels):
        result = categories[label]
        y = top + index * (bar_height + 14)
        value = float(result["percent"])
        bar_width = 6.2 * value
        lines.extend(
            [
                f'<text x="24" y="{y + 27}" font-family="Arial" font-size="15">{label.replace("_", " ").title()}</text>',
                f'<rect x="{left}" y="{y}" width="620" height="{bar_height}" rx="5" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" rx="5" fill="{colours.get(label, "#334155")}"/>',
                f'<text x="{left + 630}" y="{y + 27}" font-family="Arial" font-size="15">{result["successes"]}/{result["trials"]} ({value:.1f}%)</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top + chart_height}" x2="{left + 620}" y2="{top + chart_height}" stroke="#475569"/>',
            f'<text x="{left}" y="{top + chart_height + 25}" font-family="Arial" font-size="12">0%</text>',
            f'<text x="{left + 590}" y="{top + chart_height + 25}" font-family="Arial" font-size="12">100%</text>',
            '<text x="24" y="410" font-family="Arial" font-size="12" fill="#475569">Frozen Groq Llama 3.3 protocol; five repetitions per command; constrained in-memory execution.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials",
        type=Path,
        default=Path("evaluation/results/trials"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/analysis"),
    )
    args = parser.parse_args()
    result = analyse(args.trials)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "extended-metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (args.output / "category-performance.svg").write_text(
        category_svg(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
