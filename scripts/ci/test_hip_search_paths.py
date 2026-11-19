"""Check HIP search-path overrides and their dependency-cache identities.

Run from the repository root:
    python3 -m unittest discover -s scripts/ci -p 'test_hip_search_paths.py' -v

Tool discovery and versions are simulated, so ROCm and network access are not
required. When CMake is available, its script mode checks the effective path
after parsing each generated command's options; no GPU compilation occurs.
"""

import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import dependencies
import run


class HIPSearchPathTests(unittest.TestCase):
    def prepare(self, directory, options):
        """Prepare a real HIP context using synthetic revisions and tool identities."""
        manifest = {
            "schema_version": 1,
            "dependencies": {
                name: {"repository": repository, "revision": str(index) * 40}
                for index, (name, repository) in enumerate(dependencies.REPOSITORIES.items(), 1)
            },
        }
        path = Path(directory) / "dependencies.json"
        path.write_text(json.dumps(manifest))
        args = argparse.Namespace(configuration="hip", work_dir=directory,
                                  manifest=str(path), github_output=None)
        environment = {"GENDIL_CI_CMAKE_OPTIONS": shlex.join(options)}
        with patch.dict(os.environ, environment, clear=True), \
             patch.object(run, "tool", side_effect=lambda name: "/tools/" + Path(name).name), \
             patch.object(run, "version", return_value="test toolchain version"), \
             patch.object(run.shutil, "which", return_value=None), \
             patch.object(run, "package_inventory"), contextlib.redirect_stdout(io.StringIO()):
            run.prepare(args)
        return args, manifest, run.context_for(args)

    def test_default_path_does_not_become_a_user_override(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest, context = self.prepare(directory, [])
            self.assertEqual(context["cmake_options"], ["-DCMAKE_PREFIX_PATH=/opt/rocm"])
            self.assertEqual(context["user_cmake_options"], [])
            self.assertEqual(context["architecture"], "gfx90a")
            changed = dict(context, architecture="gfx942")
            self.assertEqual(dependencies.cache_key(manifest, "caliper", context),
                             dependencies.cache_key(manifest, "caliper", changed))

    @unittest.skipUnless(shutil.which("cmake"), "CMake is required to check option precedence")
    def test_user_search_path_is_effective_for_gendil_and_dependencies(self):
        cases = (
            ([], "/opt/rocm"),
            (["-DCMAKE_PREFIX_PATH=/custom"], "/custom"),
            (["-DCMAKE_PREFIX_PATH:STRING=/custom root;/second"], "/custom root;/second"),
            (["-DCMAKE_PREFIX_PATH=/earlier", "-DCMAKE_PREFIX_PATH:PATH=/last"], "/last"),
            (["-DCMAKE_PREFIX_PATH:STRING="], ""),
        )
        for options, expected in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                args, _, context = self.prepare(directory, options)
                self.assertEqual(context["user_cmake_options"], options)
                with patch.object(run, "find_package_dir", return_value="/test-package"):
                    gendil = run.configure_command(args, context)
                commands = {"gendil": [option for option in gendil if option.startswith("-D")]}
                commands.update({name: dependencies.cmake_options(name, context, directory)
                                 for name in dependencies.REPOSITORIES})
                # Let CMake resolve duplicate/typed -D arguments and list values.
                script = Path(directory) / "check-prefix.cmake"
                script.write_text('file(WRITE "${RESULT_FILE}" "${CMAKE_PREFIX_PATH}")\n')
                for name, definitions in commands.items():
                    with self.subTest(project=name):
                        result = Path(directory) / (name + "-prefix.txt")
                        subprocess.run(["cmake", *definitions, f"-DRESULT_FILE={result}",
                                        "-P", str(script)], check=True, capture_output=True, text=True)
                        self.assertEqual(result.read_text(), expected)

    def test_explicit_path_keeps_conservative_dependency_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest, context = self.prepare(directory, ["-DCMAKE_PREFIX_PATH=/custom"])
            self.assertEqual(dependencies.cache_context("caliper", context), context)
            changed = copy.deepcopy(context)
            changed["user_cmake_options"] = ["-DCMAKE_PREFIX_PATH=/another"]
            changed["cmake_options"][-1] = changed["user_cmake_options"][0]
            for name in dependencies.REPOSITORIES:
                self.assertNotEqual(dependencies.cache_key(manifest, name, context),
                                    dependencies.cache_key(manifest, name, changed))


if __name__ == "__main__":
    unittest.main()
