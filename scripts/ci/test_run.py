"""Check CI-driver job selection, preparation, and failure-report isolation.

Run from the repository root:
    python3 -m unittest discover -s scripts/ci -p 'test_run.py' -v

Most tests replace external tools with controlled outputs and temporary files;
they need neither GPU tools nor dependency downloads. Small Python subprocesses
exercise matrix output and command failures. Real CMake object preservation is
covered separately in test_local_reruns.py.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run


class DriverTests(unittest.TestCase):
    def args(self, directory, configuration):
        """Keep each simulated run's build, reports, and manifest in one test directory."""
        return argparse.Namespace(configuration=configuration, work_dir=directory,
                                  manifest=str(Path(directory) / "missing-manifest.json"),
                                  github_output=None, dependency=None, jobs=2)

    @staticmethod
    def write(path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    def run_check(self, args, fail=None):
        """Simulate configure/build/test outputs, optionally failing the named phase.

        Only command execution and configuration inspection are replaced; the
        driver's cleanup and phase tracking still operate on real files.
        """
        _, build, reports = run.paths(args)

        def execute(command, log, env=None):
            stage = ("test" if command[0] == "ctest" else
                     "build" if "--build" in command else "configure")
            self.write(log, f"{stage} output\n")
            if stage == "configure":
                self.write(build / "CMakeCache.txt", "current configuration")
                self.write(build / "CMakeFiles/CMakeConfigureLog.yaml", "current compiler probe")
            if stage == fail:
                raise RuntimeError(f"intentional {stage} failure")
            if stage == "test":
                self.write(reports / "junit.xml", '<testsuite failures="0"/>')
                self.write(build / "Testing/Temporary/LastTest.log", "passed tests")
                self.write(build / "configure-tests/on/CMakeCache.txt", "passed nested configure")
                self.write(build / "installed-package-test/Release/build.log", "passed consumer")

        with patch.object(run, "context_for", return_value={}), \
             patch.object(run, "configure_command", return_value=["cmake", "configure"]), \
             patch.object(run, "check_features"), patch.object(run.coverage, "collect"), \
             patch.object(run, "execute", side_effect=execute):
            run.check(args)

    def collect_report(self, args):
        """Collect the attempt's diagnostics without consulting a machine's ccache."""
        with patch.object(run.shutil, "which", return_value=None):
            run.report(args)

    def test_cuda_frontends_are_explicit(self):
        for name, compiler in (("cuda-nvcc", "NVIDIA"), ("cuda-clang", "Clang")):
            with self.subTest(configuration=name), tempfile.TemporaryDirectory() as directory:
                cfg = run.configuration(name)
                context = dict(cfg, generator="Ninja", cache_prefix=directory,
                               cmake_options=[], ccache=None)
                with patch.object(run, "find_package_dir", return_value="/tpl"):
                    command = run.configure_command(self.args(directory, name), context)
                self.assertEqual(cfg["device_compiler_id"], compiler)
                self.assertIn("-DCMAKE_CUDA_COMPILER=" + cfg["device_compiler"], command)
                self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=90", command)
                self.assertIn("-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG", command)
                self.assertIn("-DCUDAToolkit_ROOT=/usr/local/cuda", command)
                self.assertEqual(any(option.startswith("-DCMAKE_CUDA_HOST_COMPILER=")
                                     for option in command), compiler == "NVIDIA")
                for dependency in ("MFEM", "HYPRE", "CALIPER"):
                    self.assertIn("-DUSE_" + dependency + "=ON", command)

    def test_benchmarks_are_enabled_only_in_cpu_and_analysis_jobs(self):
        for name, cfg in run.CONFIG["configurations"].items():
            with self.subTest(configuration=name), tempfile.TemporaryDirectory() as directory:
                expected = cfg["backend"] == "cpu" and name != "sanitizers"
                self.assertEqual(cfg["benchmarks"], expected)
                context = dict(cfg, generator="Ninja", cache_prefix=directory,
                               cmake_options=[], ccache=None)
                with patch.object(run, "find_package_dir", return_value="/tpl"):
                    command = run.configure_command(self.args(directory, name), context)
                option = "-DGENDIL_ENABLE_BENCHMARKS=" + ("ON" if expected else "OFF")
                self.assertIn(option, command)

    def test_gpu_jobs_cannot_run_the_general_test_suite(self):
        for configuration in ("cuda-nvcc", "cuda-clang", "hip"):
            with self.subTest(configuration=configuration), tempfile.TemporaryDirectory() as directory:
                args = self.args(directory, configuration)
                _, _, reports = run.paths(args)
                reports.mkdir(parents=True)
                with patch.object(run, "context_for", return_value={}), \
                     patch.object(run, "configure_command", return_value=["cmake"]), \
                     patch.object(run, "check_features"), patch.object(run.coverage, "collect"), \
                     patch.object(run, "execute", side_effect=lambda command, log, **kwargs:
                                  self.write(log, "Skipping test example: excluded for NVCC compiler\n")) as execute:
                    run.check(args)
                commands = [call.args[0] for call in execute.call_args_list]
                ctest = next(command for command in commands if command[0] == "ctest")
                self.assertEqual(ctest[ctest.index("-L") + 1], "^(configure|compile-failure|package)$")
                self.assertIn("--no-tests=error", ctest)
                self.assertEqual(ctest[ctest.index("--timeout") + 1], "300")
                self.assertTrue((reports / "excluded-tests.txt").read_text())

    def test_cpu_jobs_keep_full_test_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "minimal")
            _, _, reports = run.paths(args)
            reports.mkdir(parents=True)
            with patch.object(run, "context_for", return_value={}), \
                 patch.object(run, "configure_command", return_value=["cmake"]), \
                 patch.object(run, "check_features"), patch.object(run.coverage, "collect"), \
                 patch.object(run, "execute", side_effect=lambda command, log, **kwargs:
                              self.write(log, "current command output\n")) as execute:
                run.check(args)
            ctest = execute.call_args_list[-1].args[0]
            self.assertEqual(ctest[0], "ctest")
            self.assertNotIn("-L", ctest)
            self.assertNotIn("-R", ctest)
            self.assertNotIn("-E", ctest)

    def test_gpu_features_cannot_silently_fall_back_to_host(self):
        features = ("GENDIL_USE_HYPRE_DEVICE", "GENDIL_HAS_DEVICE_SPARSE_FINALIZATION")
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text("USE_HIP:BOOL=ON\n")
            for disabled in features:
                (build / "GENDILConfig.cmake").write_text("\n".join(
                    f'set({name} "{"OFF" if name == disabled else "ON"}")' for name in features))
                with self.assertRaisesRegex(RuntimeError, disabled):
                    run.check_features(build, {"backend": "hip"}, build)
            (build / "GENDILConfig.cmake").write_text("\n".join(
                f'set({name} "ON")' for name in features))
            run.check_features(build, {"backend": "hip"}, build)

    def test_native_fingerprint_ignores_transient_cpu_details(self):
        first = "vendor_id : GenuineIntel\nmodel : 85\nflags : fma avx2\ncpu MHz : 2800\napicid : 0\n"
        second = first.replace("2800", "3100").replace("apicid : 0", "apicid : 8")
        self.assertEqual(run.cpu_fingerprint(first), run.cpu_fingerprint(second))
        self.assertNotEqual(run.cpu_fingerprint(first), run.cpu_fingerprint(first.replace("fma ", "")))

    def test_failure_preserves_both_output_streams(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()), \
             contextlib.redirect_stdout(io.StringIO()):
            log = Path(directory) / "failure.log"
            with self.assertRaisesRegex(RuntimeError, "Command failed"):
                run.execute([sys.executable, "-c",
                             "import sys; print('stdout diagnostic'); print('stderr diagnostic', file=sys.stderr); sys.exit(7)"], log)
            self.assertIn("stdout diagnostic", log.read_text())
            self.assertIn("stderr diagnostic", log.read_text())

    def test_report_collects_partial_configure_and_nested_failure_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "minimal")
            work, build, reports = run.paths(args)
            files = {
                build / "CMakeCache.txt": "partial configuration",
                build / "CMakeFiles/CMakeConfigureLog.yaml": "compiler probe failure",
                build / "configure-tests/on/CMakeCache.txt": "nested configuration",
                build / "installed-package-test/Release/build.log": "consumer failure",
                work / "dependency-build/gcc/hypre/configure.log": "dependency failure",
            }
            for path, contents in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
            run.set_status(reports, configure="passed", test="failed", hypre="failed")
            with patch.object(run.shutil, "which", return_value=None):
                run.report(args)
            self.assertEqual((reports / "CMakeCache.txt").read_text(), "partial configuration")
            self.assertIn("compiler probe", (reports / "CMakeConfigureLog.yaml").read_text())
            self.assertTrue((reports / "configure-probes/on/CMakeCache.txt").is_file())
            self.assertTrue((reports / "installed-consumer/Release/build.log").is_file())
            self.assertTrue((reports / "dependency-logs/hypre/configure.log").is_file())

    def test_sanitizers_only_appear_when_requested(self):
        command = [sys.executable, str(Path(run.__file__)), "matrix"]
        normal = json.loads(subprocess.check_output(command, text=True))["include"]
        scheduled = json.loads(subprocess.check_output(command + ["--include-sanitizers"], text=True))["include"]
        self.assertEqual(len(normal), 7)
        self.assertTrue({"cuda-nvcc", "cuda-clang", "hip"} <=
                        {row["configuration"] for row in normal})
        self.assertNotIn("sanitizers", {row["configuration"] for row in normal})
        self.assertEqual({row["configuration"] for row in scheduled},
                         {row["configuration"] for row in normal} | {"sanitizers"})
        self.assertTrue(all("@sha256:" in row["image"] for row in scheduled))
        self.assertTrue(all(row["tpls"] == run.configuration(row["configuration"])["tpls"]
                            for row in scheduled))

    def test_dependency_free_matrix_retains_sanitizer_schedule(self):
        command = [sys.executable, str(Path(run.__file__)), "matrix", "--dependency-free"]
        normal = json.loads(subprocess.check_output(command, text=True))["include"]
        scheduled = json.loads(subprocess.check_output(command + ["--include-sanitizers"], text=True))["include"]
        self.assertEqual([row["configuration"] for row in normal], ["minimal"])
        self.assertEqual({row["configuration"] for row in scheduled}, {"minimal", "sanitizers"})
        self.assertTrue(all(not row["tpls"] for row in scheduled))

    def test_cuda_frontends_select_independent_toolkit_images(self):
        command = [sys.executable, str(Path(run.__file__)), "matrix"]
        entries = json.loads(subprocess.check_output(command, text=True))["include"]
        images = {row["configuration"]: row["image"] for row in entries}
        self.assertTrue(images["cuda-nvcc"].startswith(
            "nvidia/cuda:13.5.0-devel-ubuntu22.04@sha256:"))
        self.assertEqual(images["cuda-clang"], run.CONFIG["images"]["cuda-clang"])
        self.assertNotEqual(images["cuda-nvcc"], images["cuda-clang"])

    def test_dependency_free_prepare_needs_no_manifest_or_network(self):
        for configuration in ("minimal", "sanitizers"):
            with self.subTest(configuration=configuration), tempfile.TemporaryDirectory() as directory:
                args = self.args(directory, configuration)
                with patch.dict(os.environ, {}, clear=True), \
                     patch.object(run, "tool", side_effect=lambda name: "/tools/" + name), \
                     patch.object(run, "version", return_value="tool version"), \
                     patch.object(run.shutil, "which", return_value=None), \
                     patch.object(run.subprocess, "check_output", side_effect=AssertionError("external request")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run.prepare(args)
                _, _, reports = run.paths(args)
                self.assertFalse(Path(args.manifest).exists())
                self.assertEqual(json.loads((reports / "dependencies.json").read_text())["dependencies"], {})
                manifest = json.loads((reports / "manifest.json").read_text())
                self.assertEqual(manifest["dependency_build_options"], {})
                self.assertEqual(run.phase_status(reports)["prepare"], "passed")

    def test_external_work_directory_keeps_its_dependency_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, work = base / "source", base / "external-build"
            with patch.object(run, "ROOT", root), \
                 patch.dict(os.environ, {"RUNNER_TEMP": str(root / "unsuitable-temp")}):
                self.assertEqual(run.dependency_prefix(work, "gcc"), work / "dependencies/gcc")
                self.assertEqual(run.dependency_prefix(work, "clang"), work / "dependencies/clang")

    def test_in_source_dependency_prefix_is_external_stable_and_profile_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, runner_temp = base / "source", base / "runner-temp"
            with patch.object(run, "ROOT", root), \
                 patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}):
                gcc = run.dependency_prefix(root / ".ci", "gcc")
                clang = run.dependency_prefix(root / ".ci", "clang")
                self.assertTrue(gcc.is_relative_to(runner_temp))
                self.assertFalse(gcc.is_relative_to(root))
                self.assertEqual(gcc.name, "gcc")
                self.assertEqual(clang.name, "clang")
                self.assertEqual(gcc.parent, clang.parent)
                self.assertEqual(run.dependency_prefix(root / ".ci", "gcc"), gcc)
                self.assertEqual(run.dependency_prefix(root / "another-build", "gcc"), gcc)
            other_root = base / "another-checkout"
            with patch.object(run, "ROOT", other_root), \
                 patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}):
                self.assertNotEqual(run.dependency_prefix(other_root / ".ci", "gcc"), gcc)

    def test_in_source_dependency_prefix_uses_local_temp_and_rejects_source_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, local_temp = base / "source", base / "local-temp"
            with patch.object(run, "ROOT", root), patch.dict(os.environ, {}, clear=True), \
                 patch.object(run.tempfile, "gettempdir", return_value=str(local_temp)):
                prefix = run.dependency_prefix(root / ".ci", "gcc")
                self.assertTrue(prefix.is_relative_to(local_temp))
                with patch.dict(os.environ, {"RUNNER_TEMP": ""}):
                    self.assertEqual(run.dependency_prefix(root / ".ci", "gcc"), prefix)
                with patch.dict(os.environ, {"RUNNER_TEMP": str(root / "temp")}):
                    with self.assertRaisesRegex(RuntimeError, "outside the GenDiL source tree"):
                        run.dependency_prefix(root / ".ci", "gcc")
                with patch.object(run.tempfile, "gettempdir", return_value=str(root / "temp")):
                    with self.assertRaisesRegex(RuntimeError, "outside the GenDiL source tree"):
                        run.dependency_prefix(root / ".ci", "gcc")

    def test_prepare_failure_cannot_republish_previous_context(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "minimal")
            work, build, reports = run.paths(args)
            for name in ("context.json", "manifest.json", "junit.xml", "build.log"):
                self.write(reports / name, "old success")
            self.write(build / "Testing/Temporary/LastTest.log", "old passed tests")
            # This sentinel tests cleanup scope, not CMake's object-file behavior.
            self.write(build / "compiled.o", "retained object")
            self.write(work / "ccache/entry", "retained compiler cache")
            with patch.object(run, "tool", side_effect=RuntimeError("compiler missing")), \
                 patch.object(run.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "compiler missing"):
                    run.prepare(args)
            self.collect_report(args)
            self.assertEqual(run.phase_status(reports)["prepare"], "failed")
            for name in ("context.json", "manifest.json", "junit.xml", "build.log", "LastTest.log"):
                self.assertFalse((reports / name).exists(), name)
            self.assertTrue((reports / "packages.json").is_file())
            self.assertTrue((build / "compiled.o").is_file())
            self.assertTrue((work / "ccache/entry").is_file())

    def test_failed_rerun_removes_old_success_reports(self):
        for failure in ("configure", "build"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                args = self.args(directory, "minimal")
                _, build, reports = run.paths(args)
                self.run_check(args)
                self.collect_report(args)
                self.assertTrue((reports / "junit.xml").is_file())
                self.assertTrue((reports / "LastTest.log").is_file())
                self.write(reports / "features.txt", "old features")
                self.write(reports / "coverage.json", "old coverage")
                inventory_files = ("ctest-inventory.json", "ctest-inventory.log",
                                   "ctest-gpu-selection.json", "ctest-gpu-selection.log")
                for name in inventory_files:
                    self.write(reports / name, "old CTest inventory")
                self.write(reports / "context.json", "preserved preparation")
                self.write(build / "compiled.o", "preserved object")
                with self.assertRaisesRegex(RuntimeError, f"intentional {failure} failure"):
                    self.run_check(args, fail=failure)
                self.collect_report(args)
                first = {str(path.relative_to(reports)): path.read_text()
                         for path in reports.rglob("*") if path.is_file()}
                self.collect_report(args)
                second = {str(path.relative_to(reports)): path.read_text()
                          for path in reports.rglob("*") if path.is_file()}
                self.assertEqual(first, second)
                for name in ("junit.xml", "LastTest.log", "features.txt", "coverage.json", "ctest.log",
                             "installed-consumer", "configure-probes") + inventory_files:
                    self.assertFalse((reports / name).exists(), name)
                self.assertEqual(run.phase_status(reports)[failure], "failed")
                self.assertEqual(run.phase_status(reports)["test"], "unattempted")
                self.assertEqual((reports / "CMakeCache.txt").read_text(), "current configuration")
                self.assertTrue((reports / "context.json").is_file())
                self.assertTrue((build / "compiled.o").is_file())

    def test_dependency_failure_invalidates_check_results(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "gcc")
            args.dependency = "hypre"
            work, _, reports = run.paths(args)
            self.run_check(args)
            self.collect_report(args)
            self.write(reports / "context.json", json.dumps({"cache_prefix": str(work / "dependencies/gcc")}))
            self.write(reports / "dependencies.json", "{}")
            dependency_work = work / "dependency-build/gcc/hypre"
            self.write(dependency_work / "build.log", "old dependency success")

            def fail(command, log, env=None):
                self.write(log, "current dependency failure")
                self.write(dependency_work / "configure.log", "current dependency configure failure")
                raise RuntimeError("dependency failed")

            with patch.object(run, "execute", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "dependency failed"):
                    run.dependency_build(args)
            self.collect_report(args)
            self.assertEqual(run.phase_status(reports)["hypre"], "failed")
            self.assertEqual(run.phase_status(reports)["test"], "unattempted")
            for name in ("junit.xml", "LastTest.log", "CMakeCache.txt", "ctest.log", "installed-consumer"):
                self.assertFalse((reports / name).exists(), name)
            self.assertEqual((reports / "dependency-logs/hypre/configure.log").read_text(),
                             "current dependency configure failure")
            self.assertFalse((reports / "dependency-logs/hypre/build.log").exists())

    def test_warm_dependency_reuse_does_not_report_old_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "gcc")
            args.dependency = "hypre"
            work, _, reports = run.paths(args)
            prefix = work / "dependencies/gcc"
            self.write(reports / "context.json", json.dumps({"cache_prefix": str(prefix)}))
            self.write(reports / "dependencies.json", "{}")
            self.write(prefix / "hypre/lib/libHYPRE.a", "preserved installed library")
            run.set_status(reports, hypre="failed")
            for relative in ("fetch.log", "configure.log", "build.log", "install.log",
                             "build/CMakeCache.txt", "build/CMakeFiles/CMakeConfigureLog.yaml"):
                self.write(work / "dependency-build/gcc/hypre" / relative, "old failure")
            self.write(work / "dependency-build/gcc/caliper/configure.log", "unattempted old failure")
            self.collect_report(args)
            with patch.object(run, "execute", side_effect=lambda command, log, **kwargs:
                              self.write(log, "Reusing verified hypre installation\n")):
                run.dependency_build(args)
            self.collect_report(args)
            self.assertEqual(run.phase_status(reports)["hypre"], "passed")
            self.assertIn("Reusing verified", (reports / "dependency-hypre.log").read_text())
            self.assertFalse((reports / "dependency-logs/hypre").exists())
            self.assertFalse((reports / "dependency-logs/caliper").exists())
            self.assertTrue((prefix / "hypre/lib/libHYPRE.a").is_file())

    def test_package_inventory_is_sorted_and_excludes_removed_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            output = "zlib\t1.3\tamd64\tinstalled\nremoved\t1\tall\tconfig-files\ncmake\t3.30.5\tamd64\tinstalled\n"
            with patch.object(run.shutil, "which", return_value="/usr/bin/dpkg-query"), \
                 patch.object(run.subprocess, "check_output", return_value=output), \
                 patch.object(run.metadata, "version", return_value="test version"):
                run.package_inventory(reports)
            inventory = json.loads((reports / "packages.json").read_text())
            self.assertEqual(inventory["apt_packages"], [["cmake", "3.30.5", "amd64"], ["zlib", "1.3", "amd64"]])
            self.assertEqual(inventory["python_build_tools"]["cmake"], "test version")

    def test_package_inventory_can_be_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory)
            with patch.object(run.shutil, "which", return_value=None):
                run.package_inventory(reports)
            self.assertIn("not available", json.loads((reports / "packages.json").read_text())["apt_unavailable"])
            with patch.object(run.shutil, "which", return_value="/usr/bin/dpkg-query"), \
                 patch.object(run.subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "dpkg-query")):
                run.package_inventory(reports)
            self.assertIn("apt_unavailable", json.loads((reports / "packages.json").read_text()))


if __name__ == "__main__":
    unittest.main()
