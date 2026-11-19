"""Check that coverage describes configured targets and enforces GPU boundaries."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import ci_coverage as coverage


class CoverageTests(unittest.TestCase):
    def fixture(self, directory, language="CUDA", compiler="NVIDIA", benchmarks=True):
        build, reports = Path(directory) / "build", Path(directory) / "reports"
        reply = build / ".cmake/api/v1/reply"
        reply.mkdir(parents=True)
        reports.mkdir()
        objects = [{"kind": kind, "version": {"major": version}, "jsonFile": kind + ".json"}
                   for kind, version in (("codemodel", 2), ("toolchains", 1))]
        (reply / "index-example.json").write_text(json.dumps({"objects": objects}))
        targets = []
        for category, name in (("tests", "test-example"), ("examples", "example"),
                               ("tests", coverage.EXPERIMENT), ("benchmarks", "benchmark")):
            if category == "benchmarks" and not benchmarks:
                continue
            filename = name + ".json"
            target = {"name": name, "type": "EXECUTABLE", "paths": {"source": category},
                      "compileGroups": [{"language": language}],
                      "sources": [{"path": category + "/" + name + ".cpp", "compileGroupIndex": 0},
                                  {"path": "header.hpp"}]}
            (reply / filename).write_text(json.dumps(target))
            targets.append({"jsonFile": filename})
        model = {"paths": {"source": "/source"},
                 "configurations": [{"name": "Release", "targets": targets}]}
        (reply / "codemodel.json").write_text(json.dumps(model))
        (reply / "toolchains.json").write_text(json.dumps({"toolchains": [
            {"language": language, "compiler": {"id": compiler, "version": "1.2", "path": "/compiler"}}]}))
        (reports / "configure.log").write_text(
            "-- Skipping test test-unsupported: excluded for NVCC compiler\n"
            "-- Skipping benchmark benchmark-unsupported: excluded for CUDA backend\n")
        return build, reports

    def ctest(self, command, **kwargs):
        names = [("configure-example", "configure"), ("test-installed-package", "package")]
        if "-L" not in command:
            names += [("test-example", "numerical")]
        data = {"tests": [{"name": name, "properties": [{"name": "LABELS", "value": [label]}]}
                          for name, label in names]}
        return subprocess.CompletedProcess(command, 0, json.dumps(data), "")

    def test_queries_precede_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            coverage.request(directory)
            query = Path(directory) / ".cmake/api/v1/query/client-gendil-ci"
            self.assertEqual({item.name for item in query.iterdir()}, {"codemodel-v2", "toolchains-v1"})

    def test_gpu_inventory_keeps_configured_and_executed_scope_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            build, reports = self.fixture(directory)
            with patch.object(coverage.subprocess, "run", side_effect=self.ctest) as invoke:
                result = coverage.collect(build, reports, {"backend": "cuda", "device_compiler_id": "NVIDIA",
                                                          "benchmarks": True})
            self.assertTrue(all("--show-only=json-v1" in call.args[0] for call in invoke.call_args_list))
            self.assertEqual(invoke.call_args_list[-1].args[0][-2:], ["-L", coverage.SAFE_TEST_LABELS])
            self.assertEqual(len(result["ctest_registered"]), 3)
            self.assertEqual(len(result["ctest_gpu_host_only_selection"]), 2)
            self.assertEqual(result["targets"]["examples"][0]["languages"], ["CUDA"])
            self.assertEqual(len(result["exclusions"]), 2)
            self.assertIn("excluded from the normal build", (reports / "coverage.txt").read_text())
            self.assertEqual(json.loads((reports / "coverage.json").read_text()), result)

    def test_wrong_device_compiler_or_language_fails_with_diagnostics(self):
        for language, actual_id, expected_text in (("CUDA", "NVIDIA", "Expected CUDA compiler Clang"),
                                                   ("CXX", "Clang", "GPU target")):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                build, reports = self.fixture(directory, language, actual_id)
                with patch.object(coverage.subprocess, "run", side_effect=self.ctest), \
                     self.assertRaisesRegex(RuntimeError, expected_text):
                    coverage.collect(build, reports, {"backend": "cuda", "device_compiler_id": "Clang"})
                self.assertTrue(json.loads((reports / "coverage.json").read_text())["errors"])

    def test_benchmarks_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory() as directory:
            build, reports = self.fixture(directory, language="CXX", compiler="GNU", benchmarks=False)
            with patch.object(coverage.subprocess, "run", side_effect=self.ctest):
                with self.assertRaisesRegex(RuntimeError, "No benchmarks"):
                    coverage.collect(build, reports, {"backend": "cpu", "benchmarks": True})
                result = coverage.collect(build, reports, {"backend": "cpu", "benchmarks": False})
            self.assertIsNone(result["ctest_gpu_host_only_selection"])

    def test_ctest_discovery_failures_preserve_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            build, reports = self.fixture(directory)
            with patch.object(coverage.subprocess, "run", return_value=subprocess.CompletedProcess(
                    ["ctest"], 1, "partial inventory", "discovery failed")), \
                 self.assertRaisesRegex(RuntimeError, "CTest inventory failed"):
                coverage.collect(build, reports, {"backend": "cuda"})
            self.assertEqual((reports / "ctest-inventory.log").read_text(), "discovery failed")


if __name__ == "__main__":
    unittest.main()
