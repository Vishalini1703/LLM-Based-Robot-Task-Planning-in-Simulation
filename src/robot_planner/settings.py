"""Configuration loading that keeps API credentials outside source control."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


class SettingsError(ValueError):
    """Raised when required runtime configuration is missing or malformed."""


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse the small KEY=VALUE subset used by dotenv files."""

    env_path = Path(path)
    if not env_path.exists():
        raise SettingsError(f"Environment file does not exist: {env_path}")

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise SettingsError(f"Invalid environment entry at {env_path}:{line_number}.")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise SettingsError(f"Invalid environment key at {env_path}:{line_number}.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def groq_api_key(
    env_file: str | Path = ".env", environ: Mapping[str, str] | None = None
) -> str:
    """Load the Groq key, accepting the project's existing GROQ_AI_KEY alias."""

    process_env = os.environ if environ is None else environ
    key = process_env.get("GROQ_API_KEY") or process_env.get("GROQ_AI_KEY")
    if not key:
        file_values = load_env_file(env_file)
        key = file_values.get("GROQ_API_KEY") or file_values.get("GROQ_AI_KEY")
    if not key:
        raise SettingsError(
            "Groq API key is missing. Set GROQ_API_KEY or GROQ_AI_KEY in the environment or .env."
        )
    return key

