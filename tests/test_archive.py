import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.build_repro_archive import audit_files, build_archive, included_files


class ArchiveTests(unittest.TestCase):
    def test_excludes_secrets_caches_and_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
            (root / ".env").write_text(
                "GROQ" + "_AI_KEY=" + "g" + "sk_secretvalue",
                encoding="utf-8",
            )
            (root / "35043637_Chapter_1_and_2.docx").write_bytes(b"private")
            (root / "webots" / "runtime").mkdir(parents=True)
            (root / "webots" / "runtime" / "run.json").write_text("{}", encoding="utf-8")
            files = included_files(root)
            relatives = {path.relative_to(root).as_posix() for path in files}
            self.assertEqual(relatives, {"src/app.py"})

    def test_audit_rejects_machine_path_and_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "bad.md"
            path.write_text(
                "GROQ" + "_API_KEY=" + "g" + "sk_" + "abcdefghijklmnop\n"
                + "F:" + "\\client\\project",
                encoding="utf-8",
            )
            problems = audit_files(root, [path])
            self.assertEqual(len(problems), 3)

    def test_builds_valid_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("portable", encoding="utf-8")
            destination = root.parent / f"{root.name}.zip"
            self.addCleanup(destination.unlink, missing_ok=True)
            count, size = build_archive(root, destination)
            self.assertEqual(count, 1)
            self.assertGreater(size, 0)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.namelist(), ["README.md"])
