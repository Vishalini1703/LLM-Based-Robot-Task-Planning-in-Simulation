import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.build_submission_archive import (
    ARCHIVE_ROOT,
    build_submission,
    collect_submission_files,
)


class SubmissionArchiveTests(unittest.TestCase):
    def test_allowlist_excludes_research_and_runtime_material(self):
        root = Path(__file__).resolve().parents[1]
        names = {
            path.relative_to(root).as_posix()
            for path in collect_submission_files(root)
        }
        self.assertIn("README.md", names)
        self.assertIn("src/robot_planner/pipeline.py", names)
        self.assertIn("webots/worlds/kitchen_llm.wbt", names)
        self.assertIn("docker/webots-cnn.Dockerfile", names)
        self.assertIn(
            "cnn_artifacts/kitchen_object_net_v1/KitchenObjectNet.onnx", names
        )
        self.assertIn(
            "cnn_artifacts/kitchen_object_net_v1/verification_fixtures/mug.jpg",
            names,
        )
        self.assertIn(
            "cnn_artifacts/kitchen_object_net_v1/plots/training_curves.png", names
        )
        self.assertNotIn(".env", names)
        self.assertNotIn("plan.md", names)
        self.assertNotIn("35043637_Chapter_1_and_2.docx", names)
        self.assertFalse(any(name.startswith("evaluation/results/") for name in names))
        self.assertFalse(any(name.startswith("evaluation/feedback/") for name in names))
        self.assertFalse(any(name.startswith("webots/runtime/") for name in names))
        self.assertFalse(
            any(
                name.startswith("cnn_artifacts/kitchen_object_net_v1/dataset/")
                for name in names
            )
        )
        self.assertFalse(
            any(
                name.startswith("cnn_artifacts/kitchen_object_net_v1/checkpoints/")
                for name in names
            )
        )
        self.assertFalse(
            any(
                name.startswith("cnn_artifacts/kitchen_object_net_v1/logs/")
                for name in names
            )
        )

    def test_builds_rooted_zip_with_manifest(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "submission.zip"
            result = build_submission(root, destination)
            self.assertGreater(result["files"], 1)
            with zipfile.ZipFile(destination) as archive:
                self.assertIsNone(archive.testzip())
                names = archive.namelist()
                self.assertTrue(all(name.startswith(f"{ARCHIVE_ROOT}/") for name in names))
                manifest = json.loads(
                    archive.read(
                        f"{ARCHIVE_ROOT}/SUBMISSION_MANIFEST.json"
                    )
                )
                self.assertEqual(manifest["archive_profile"], "code_only_submission")
                self.assertNotIn(f"{ARCHIVE_ROOT}/.env", names)
                self.assertIn(
                    f"{ARCHIVE_ROOT}/cnn_artifacts/kitchen_object_net_v1/KitchenObjectNet.onnx",
                    names,
                )
