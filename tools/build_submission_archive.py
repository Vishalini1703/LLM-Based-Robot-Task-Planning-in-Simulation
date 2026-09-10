"""Build the code-only, ready-to-share project submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


ARCHIVE_ROOT = "robot-task-planner"
ROOT_FILES = {
    ".env.example",
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "setup.ps1",
    "setup.sh",
}
TECHNICAL_DOCS = {
    "architecture.md",
    "cnn-perception.md",
    "evaluation-protocol.md",
    "limitations.md",
    "plan-contract.md",
    "reproducibility.md",
}
SOURCE_DIRECTORIES = {
    "config",
    "docker",
    "examples",
    "schemas",
    "src",
    "tests",
}
WEBOTS_DIRECTORIES = {
    Path("webots/controllers"),
    Path("webots/worlds"),
}
EVALUATION_FILES = {
    Path("evaluation/commands.json"),
    Path("evaluation/frozen-settings.json"),
    Path("evaluation/protocol.json"),
}
TOOL_FILES = {
    Path("tools/build_repro_archive.py"),
    Path("tools/build_submission_archive.py"),
    Path("tools/extended_analysis.py"),
}
CNN_ROOT = Path("cnn_artifacts/kitchen_object_net_v1")
CNN_ARTIFACT_FILES = {
    CNN_ROOT / "KitchenObjectNet.onnx",
    CNN_ROOT / "MODEL_CARD.md",
    CNN_ROOT / "architecture.json",
    CNN_ROOT / "class_map.json",
    CNN_ROOT / "dataset_manifest.json",
    CNN_ROOT / "e2e_summary.json",
    CNN_ROOT / "evaluation.json",
    CNN_ROOT / "metrics.jsonl",
    CNN_ROOT / "perception_180.json",
    CNN_ROOT / "provenance.json",
    CNN_ROOT / "training_config.json",
    CNN_ROOT / "verification.json",
}
CNN_PLOTS = {
    CNN_ROOT / "plots" / name
    for name in {
        "confusion_matrix.png",
        "dataset_distribution.png",
        "difficulty_metrics.png",
        "per_class_metrics.png",
        "perception_180.png",
        "sample_predictions.png",
        "threshold_selection.png",
        "training_curves.png",
    }
}
CNN_FIXTURE_ROOT = CNN_ROOT / "verification_fixtures"
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {
    "",
    ".example",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".wbt",
}
SECRET_PATTERNS = (
    re.compile(r"\bgsk_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)(groq_ai_key|groq_api_key)\s*=\s*[^\s\"']+"),
)
ABSOLUTE_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\\\r\n]+\\)+[^\\\r\n]*")


def _allowed(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in IGNORED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return False
    if len(relative.parts) == 1:
        return relative.as_posix() in ROOT_FILES
    if relative.parts[0] == "docs":
        return relative.name in TECHNICAL_DOCS and len(relative.parts) == 2
    if relative.parts[0] in SOURCE_DIRECTORIES:
        return True
    if any(relative == directory or directory in relative.parents for directory in WEBOTS_DIRECTORIES):
        return True
    if relative in EVALUATION_FILES or relative in TOOL_FILES:
        return True
    if relative in CNN_ARTIFACT_FILES or relative in CNN_PLOTS:
        return True
    if CNN_FIXTURE_ROOT in relative.parents:
        return True
    return False


def collect_submission_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and _allowed(path, root)
    ]
    relative_files = {path.relative_to(root).as_posix() for path in files}
    required = ROOT_FILES | {
        f"docs/{name}" for name in TECHNICAL_DOCS
    } | {
        path.as_posix() for path in EVALUATION_FILES | TOOL_FILES
    } | {path.as_posix() for path in CNN_ARTIFACT_FILES | CNN_PLOTS} | {
        (CNN_FIXTURE_ROOT / "manifest.json").as_posix()
    }
    missing = sorted(required - relative_files)
    if missing:
        raise FileNotFoundError(
            "Required submission files are missing: " + ", ".join(missing)
        )
    return sorted(files)


def audit_submission(root: Path, files: list[Path]) -> None:
    problems = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                problems.append(f"{relative}: possible API credential")
        if ABSOLUTE_WINDOWS_PATH.search(content):
            problems.append(f"{relative}: machine-specific absolute Windows path")
    if problems:
        raise RuntimeError(
            "Code-submission privacy audit failed:\n" + "\n".join(problems)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_submission(root: Path, destination: Path) -> dict[str, object]:
    files = collect_submission_files(root)
    audit_submission(root, files)
    manifest = {
        "archive_profile": "code_only_submission",
        "root_directory": ARCHIVE_ROOT,
        "file_count_excluding_manifest": len(files),
        "excluded_content": [
            "API key and .env",
            "thesis/proposal DOCX",
            "plan.md",
            "participant-study materials and responses",
            "raw evaluation results and pilot data",
            "large raw CNN dataset and training checkpoints",
            "full runtime logs and screenshots (compact hashed summaries retained)",
            "caches, virtual environments, and build output",
        ],
        "files": {
            path.relative_to(root).as_posix(): _sha256(path) for path in files
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive.write(path, f"{ARCHIVE_ROOT}/{relative}")
        archive.writestr(
            f"{ARCHIVE_ROOT}/SUBMISSION_MANIFEST.json",
            json.dumps(manifest, indent=2) + "\n",
        )
    with zipfile.ZipFile(destination) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP integrity check failed at {bad_file}.")
        archived_names = set(archive.namelist())
        forbidden = {
            f"{ARCHIVE_ROOT}/.env",
            f"{ARCHIVE_ROOT}/plan.md",
            f"{ARCHIVE_ROOT}/35043637_Chapter_1_and_2.docx",
        }
        leaked = sorted(forbidden & archived_names)
        if leaked:
            raise RuntimeError("Forbidden submission files found: " + ", ".join(leaked))
    return {
        "archive": str(destination),
        "project_root_in_zip": ARCHIVE_ROOT,
        "files": len(files) + 1,
        "bytes": destination.stat().st_size,
        "manifest": f"{ARCHIVE_ROOT}/SUBMISSION_MANIFEST.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission/robot-task-planner-code-submission.zip"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    destination = (
        args.output if args.output.is_absolute() else root / args.output
    ).resolve()
    print(json.dumps(build_submission(root, destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
