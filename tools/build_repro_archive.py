"""Build a portable, secret-free reproduction ZIP from the project."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


EXCLUDED_NAMES = {
    ".env",
    ".git",
    ".idea",
    ".pytest_cache",
    ".venv",
    "35043637_Chapter_1_and_2.docx",
    "__pycache__",
    "dist",
}
EXCLUDED_PREFIXES = (
    Path("logs/runs"),
    Path("webots/runtime"),
)
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
SECRET_PATTERNS = (
    re.compile(r"\bgsk_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)(groq_ai_key|groq_api_key)\s*=\s*[^\s\"']+"),
)
ABSOLUTE_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\(?:[^\\\r\n]+\\)+[^\\\r\n]*")
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".wbt",
    ".yaml",
    ".yml",
}


def included_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_NAMES for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if any(
            relative == prefix or prefix in relative.parents
            for prefix in EXCLUDED_PREFIXES
        ):
            continue
        files.append(path)
    return sorted(files)


def audit_files(root: Path, files: list[Path]) -> list[str]:
    problems = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                problems.append(f"{relative}: possible API credential")
        if ABSOLUTE_WINDOWS_PATH.search(content):
            problems.append(f"{relative}: machine-specific absolute Windows path")
    return problems


def build_archive(root: Path, destination: Path) -> tuple[int, int]:
    files = included_files(root)
    problems = audit_files(root, files)
    if problems:
        raise RuntimeError("Archive privacy audit failed:\n" + "\n".join(problems))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(destination) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP integrity check failed at {bad_file}.")
        return len(archive.infolist()), destination.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/robot-task-planner-repro.zip"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    destination = (
        args.output
        if args.output.is_absolute()
        else (root / args.output)
    ).resolve()
    count, size = build_archive(root, destination)
    print(f"archive={destination}")
    print(f"files={count}")
    print(f"bytes={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
