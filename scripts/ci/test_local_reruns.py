"""Verify local reruns preserve objects while refreshing probes and diagnostics.

Run from the repository root:
    python3 -m unittest discover -s scripts/ci -p 'test_local_reruns.py' -v

These tests run the driver against a small temporary C++ project using real
CMake and CTest, with ccache disabled. They require CMake 3.25 or newer, cc/c++,
and Ninja or Make; missing executables skip the suite. No GenDiL dependencies
or GPU toolchain are needed.
"""

import argparse
import contextlib
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import run


@unittest.skipUnless(all(shutil.which(tool) for tool in ("cmake", "ctest", "cc", "c++"))
                     and (shutil.which("ninja") or shutil.which("make")),
                     "CMake and a host C++ toolchain are required")
class LocalRerunTests(unittest.TestCase):
    def setUp(self):
        """Substitute a small source tree while retaining the driver's full check path."""
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = directory / "source"
        self.source.mkdir()
        self.args = argparse.Namespace(configuration="minimal", work_dir=str(directory / "work"),
                                       manifest=str(directory / "unused.json"), github_output=None,
                                       dependency=None, jobs=2)
        _, self.build, self.reports = run.paths(self.args)
        # Cover root and subdirectory objects; category targets also exercise
        # the driver's normal configured-target inventory checks.
        self.project = '''cmake_minimum_required(VERSION 3.25)
project(CIRerunFixture LANGUAGES CXX)
include(CheckCXXSourceCompiles)
file(READ "${CMAKE_SOURCE_DIR}/probe.cpp" probe_source)
check_cxx_source_compiles("${probe_source}" GENDIL_CI_FIXTURE_CAPABILITY)
file(WRITE "${CMAKE_BINARY_DIR}/GENDILConfig.cmake" "# Fixture configuration\\n")
enable_testing()
add_executable(root-probe root.cpp)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(benchmarks)
'''
        (self.source / "CMakeLists.txt").write_text(self.project)
        (self.source / "probe.cpp").write_text("int main() { return 0; }\n")
        (self.source / "root.cpp").write_text("int main() { return 0; }\n")
        for category in ("tests", "examples", "benchmarks"):
            path = self.source / category
            path.mkdir()
            (path / "main.cpp").write_text("int main() { return 0; }\n")
            (path / "CMakeLists.txt").write_text(
                f"add_executable({category}-probe main.cpp)\n"
                + (f"add_test(NAME probe COMMAND {category}-probe)\n" if category == "tests" else ""))
        stack.enter_context(patch.object(run, "ROOT", self.source))
        stack.enter_context(patch.dict(os.environ, {
            "CC": shutil.which("cc"), "CXX": shutil.which("c++"),
            "CMAKE_GENERATOR": "Ninja" if shutil.which("ninja") else "Unix Makefiles",
            "GENDIL_CI_CMAKE_OPTIONS": "",
        }))
        find_tool = shutil.which
        stack.enter_context(patch.object(run.shutil, "which", side_effect=lambda name:
                                        None if name == "ccache" else find_tool(name)))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        run.prepare(self.args)

    def objects(self):
        """Record actual target-object timestamps so a rebuild cannot pass unnoticed."""
        return {path.relative_to(self.build): path.stat().st_mtime_ns
                for path in self.build.rglob("*.o") if ".dir" in str(path)}

    def test_unchanged_prepare_and_check_preserve_objects_and_refresh_probes(self):
        run.check(self.args)
        before = self.objects()
        self.assertEqual(len(before), 4)
        self.assertIn("GENDIL_CI_FIXTURE_CAPABILITY:INTERNAL=1",
                      (self.build / "CMakeCache.txt").read_text())

        # A new attempt clears reports, while a changed probe must not reuse its old result.
        (self.source / "probe.cpp").write_text("#error capability no longer available\n")
        run.prepare(self.args)
        run.check(self.args)
        self.assertEqual(self.objects(), before)
        self.assertNotIn("GENDIL_CI_FIXTURE_CAPABILITY:INTERNAL=1",
                         (self.build / "CMakeCache.txt").read_text())
        self.assertIn("-O3 -DNDEBUG", (self.build / "compile_commands.json").read_text())
        self.assertEqual(run.phase_status(self.reports)["test"], "passed")

    def test_failed_configuration_cannot_reuse_previous_success(self):
        run.check(self.args)
        before = self.objects()
        (self.source / "CMakeLists.txt").write_text(self.project + '\nmessage(FATAL_ERROR "fixture failure")\n')
        with self.assertRaisesRegex(RuntimeError, "Command failed"):
            run.check(self.args)
        run.report(self.args)
        self.assertFalse((self.reports / "junit.xml").exists())
        self.assertEqual(run.phase_status(self.reports)["configure"], "failed")
        (self.source / "CMakeLists.txt").write_text(self.project)
        run.check(self.args)
        self.assertEqual(self.objects(), before)

    def test_failed_build_retry_keeps_unaffected_objects(self):
        run.check(self.args)
        before = self.objects()
        source = self.source / "tests/main.cpp"
        # Make variants may compare timestamps only to the second.
        for path in before:
            if path.parts[0] == "tests":
                os.utime(self.build / path, (1, 1))
        source.write_text("#error fixture build failure\n")
        with self.assertRaisesRegex(RuntimeError, "Command failed"):
            run.check(self.args)
        self.assertEqual(run.phase_status(self.reports)["build"], "failed")
        source.write_text("int main() { return 0; }\n")
        run.check(self.args)
        after = self.objects()
        for path, modified in before.items():
            if path.parts[0] != "tests":
                self.assertEqual(after[path], modified)


if __name__ == "__main__":
    unittest.main()
