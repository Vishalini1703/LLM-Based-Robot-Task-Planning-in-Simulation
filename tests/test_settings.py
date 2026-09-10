from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_planner.settings import SettingsError, groq_api_key, load_env_file


class SettingsTests(unittest.TestCase):
    def test_loads_quoted_values_and_existing_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("GROQ_AI_KEY='secret-value'\n", encoding="utf-8")
            self.assertEqual({"GROQ_AI_KEY": "secret-value"}, load_env_file(path))
            self.assertEqual("secret-value", groq_api_key(path, environ={}))

    def test_process_environment_takes_precedence(self) -> None:
        self.assertEqual(
            "process-key",
            groq_api_key("does-not-need-to-exist", {"GROQ_API_KEY": "process-key"}),
        )

    def test_rejects_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OTHER=value\n", encoding="utf-8")
            with self.assertRaises(SettingsError):
                groq_api_key(path, environ={})


if __name__ == "__main__":
    unittest.main()
