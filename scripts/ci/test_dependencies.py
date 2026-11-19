"""Regression checks for dependency identity, restoration, and diagnostics."""

import copy
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import dependencies


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema_version": 1,
            "dependencies": {
                name: {"repository": repository, "ref": "example",
                       "version": "example", "revision": str(index) * 40}
                for index, (name, repository) in enumerate(dependencies.REPOSITORIES.items(), 1)
            },
        }
        self.context = {"backend": "cpu", "cc": "/usr/bin/gcc-14",
                        "cxx": "/usr/bin/g++-14", "device_compiler": None,
                        "architecture": None, "openmp": True,
                        "container": "ubuntu@sha256:example",
                        "toolchain": {"cxx": "gcc 14.2.0", "cmake": "3.30.5"},
                        "cache_prefix": "/tmp/gendil-deps", "generator": "Ninja",
                        "ccache": "/usr/bin/ccache"}

    def test_latest_resolution_uses_one_shared_manifest(self):
        responses = {
            "repos/mfem/mfem": {"default_branch": "master"},
            "repos/hypre-space/hypre/releases/latest": {"tag_name": "v3.2.1"},
            "repos/llnl/Caliper/releases/latest": {"tag_name": "v2.15.0"},
            "repos/mfem/mfem/commits/master": {"sha": "a" * 40},
            "repos/hypre-space/hypre/commits/v3.2.1": {"sha": "b" * 40},
            "repos/llnl/Caliper/commits/v2.15.0": {"sha": "c" * 40},
        }
        with patch.object(dependencies, "github_json", side_effect=responses.__getitem__) as fetch:
            result = dependencies.resolve_manifest({})
        self.assertEqual(fetch.call_count, 6)
        self.assertEqual(result["dependencies"]["mfem"]["revision"], "a" * 40)
        self.assertEqual(result["dependencies"]["hypre"]["version"], "v3.2.1")

    def test_explicit_refs_bypass_release_discovery(self):
        with patch.object(dependencies, "github_json", return_value={"sha": "a" * 40}) as fetch:
            result = dependencies.resolve_manifest({name: "releases/test" for name in dependencies.REPOSITORIES})
        self.assertEqual(fetch.call_count, 3)
        self.assertTrue(all("/commits/releases%2Ftest" in call.args[0] for call in fetch.call_args_list))
        self.assertEqual(result["dependencies"]["caliper"]["ref"], "releases/test")

    def test_prerelease_is_not_an_implicit_stable_dependency(self):
        def response(path):
            if path == "repos/mfem/mfem":
                return {"default_branch": "master"}
            if "/commits/" in path:
                return {"sha": "a" * 40}
            return {"tag_name": "v-next", "prerelease": True}
        with patch.object(dependencies, "github_json", side_effect=response):
            with self.assertRaisesRegex(RuntimeError, "not stable"):
                dependencies.resolve_manifest({})

    def test_mfem_update_does_not_invalidate_other_libraries(self):
        changed = copy.deepcopy(self.manifest)
        changed["dependencies"]["mfem"]["revision"] = "f" * 40
        changed["resolved_at"] = "different workflow start"
        for name in ("hypre", "caliper"):
            self.assertEqual(dependencies.cache_key(self.manifest, name, self.context),
                             dependencies.cache_key(changed, name, self.context))
        self.assertNotEqual(dependencies.cache_key(self.manifest, "mfem", self.context),
                            dependencies.cache_key(changed, "mfem", self.context))

    def test_toolchain_and_configuration_changes_invalidate_cache(self):
        key = dependencies.cache_key(self.manifest, "hypre", self.context)
        for field, value in (("openmp", False), ("container", "another-image"),
                             ("cache_prefix", "/another-install"),
                             ("cmake_options", ["-DHYPRE_ENABLE_PRINT_ERRORS=ON"]),
                             ("toolchain", {"cxx": "gcc 15.2.0"})):
            with self.subTest(field=field):
                changed = dict(self.context, **{field: value})
                self.assertNotEqual(key, dependencies.cache_key(self.manifest, "hypre", changed))

    def test_explicit_device_architecture_is_required(self):
        for architecture in (None, "native"):
            context = dict(self.context, backend="hip", device_compiler="amdclang++",
                           architecture=architecture)
            with self.assertRaises(ValueError):
                dependencies.cache_key(self.manifest, "hypre", context)

    def test_recipes_preserve_backend_and_portable_flags(self):
        for backend, architecture in (("cpu", None), ("cuda", "90"), ("hip", "gfx90a")):
            context = dict(self.context, backend=backend, device_compiler="compiler",
                           architecture=architecture)
            for name in dependencies.REPOSITORIES:
                options = dependencies.cmake_options(name, context, "/prefix")
                self.assertIn("-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG", options)
                self.assertNotIn("native", " ".join(options))
                if name == "mfem":
                    self.assertIn("-DMFEM_PRECISION=double", options)
                    self.assertIn("-DCMAKE_CXX_SCAN_FOR_MODULES=OFF", options)
                else:
                    self.assertNotIn("-DCMAKE_CXX_SCAN_FOR_MODULES=OFF", options)
                if name == "caliper":
                    self.assertNotIn("-DCMAKE_HIP_COMPILER=compiler", options)
                    self.assertNotIn("-DCMAKE_CUDA_COMPILER=compiler", options)
                elif backend != "cpu":
                    self.assertIn(f"-DCMAKE_{backend.upper()}_ARCHITECTURES={architecture}", options)

    def test_hip_wrapper_never_requires_device_discovery(self):
        context = dict(self.context, backend="hip", cxx="/opt/rocm/bin/hipcc",
                       host_cxx="/usr/bin/g++-12", device_compiler="/opt/rocm/llvm/bin/clang++",
                       architecture="gfx90a")
        for name in ("mfem", "hypre"):
            options = dependencies.cmake_options(name, context, "/prefix")
            self.assertIn("-DCMAKE_CXX_FLAGS=--offload-arch=gfx90a", options)
            self.assertIn("-DHIP_HIPCC_FLAGS=--offload-arch=gfx90a", options)
            self.assertIn("-DAMDGPU_TARGETS=gfx90a", options)
        caliper = dependencies.cmake_options("caliper", context, "/prefix")
        self.assertIn("-DCMAKE_CXX_COMPILER=/usr/bin/g++-12", caliper)
        self.assertIn("-DCMAKE_CXX_FLAGS=", caliper)

    def test_cuda_dependencies_use_the_selected_compiler_and_toolkit(self):
        for compiler_id, compiler, cxx in (
                ("NVIDIA", "/usr/local/cuda/bin/nvcc", "/usr/bin/g++-12"),
                ("Clang", "/usr/bin/clang++-18", "/usr/bin/clang++-18")):
            context = dict(self.context, backend="cuda", cxx=cxx,
                           device_compiler_id=compiler_id, device_compiler=compiler,
                           architecture="90", cuda_toolkit_root="/usr/local/cuda")
            for name in ("mfem", "hypre"):
                with self.subTest(compiler=compiler_id, dependency=name):
                    options = dependencies.cmake_options(name, context, "/prefix")
                    self.assertIn(f"-DCMAKE_CUDA_COMPILER={compiler}", options)
                    self.assertIn("-DCUDAToolkit_ROOT=/usr/local/cuda", options)
                    self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=90", options)
                    self.assertIn("-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG", options)
                    self.assertEqual(f"-DCMAKE_CUDA_HOST_COMPILER={cxx}" in options,
                                     compiler_id == "NVIDIA")
            caliper = dependencies.cmake_options("caliper", context, "/prefix")
            self.assertFalse(any(option.startswith("-DCMAKE_CUDA_") for option in caliper))
            self.assertNotIn("-DCUDAToolkit_ROOT=/usr/local/cuda", caliper)

    def test_cuda_compiler_identity_and_toolkit_invalidate_dependency_caches(self):
        context = dict(self.context, backend="cuda", device_compiler="/usr/bin/cuda-cxx",
                       device_compiler_id="NVIDIA", architecture="90",
                       cuda_toolkit_root="/usr/local/cuda")
        for name in ("mfem", "hypre"):
            key = dependencies.cache_key(self.manifest, name, context)
            for field, value in (("device_compiler_id", "Clang"),
                                 ("cuda_toolkit_root", "/opt/cuda")):
                with self.subTest(dependency=name, field=field):
                    changed = dict(context, **{field: value})
                    self.assertNotEqual(key, dependencies.cache_key(self.manifest, name, changed))

    def test_host_caliper_ignores_unused_device_metadata(self):
        context = dict(self.context, backend="cuda", device_compiler="/cuda/nvcc",
                       device_compiler_id="NVIDIA", architecture="90",
                       cuda_toolkit_root="/cuda", user_cmake_options=[], cmake_options=[])
        context["toolchain"] = dict(context["toolchain"], device="NVCC 12.2",
                                    cuda_toolkit="CUDA 12.2")
        key = dependencies.cache_key(self.manifest, "caliper", context)
        changes = [dict(context, architecture="100"),
                   dict(context, device_compiler="/cuda/new/nvcc"),
                   dict(context, device_compiler_id="Clang"),
                   dict(context, cuda_toolkit_root="/cuda/new"),
                   dict(context, toolchain=dict(context["toolchain"],
                                                device="NVCC 13.1", cuda_toolkit="CUDA 13.1"))]
        for changed in changes:
            self.assertEqual(key, dependencies.cache_key(self.manifest, "caliper", changed))
            for name in ("mfem", "hypre"):
                self.assertNotEqual(dependencies.cache_key(self.manifest, name, context),
                                    dependencies.cache_key(self.manifest, name, changed))

    def test_caliper_identity_keeps_real_host_build_inputs(self):
        context = dict(self.context, user_cmake_options=[], cmake_options=[])
        key = dependencies.cache_key(self.manifest, "caliper", context)
        for field, value in (("cc", "/different/gcc"), ("cxx", "/different/g++"),
                             ("container", "changed-image"), ("cache_prefix", "/different/prefix"),
                             ("generator", "Unix Makefiles"), ("ccache", "/different/ccache"),
                             ("cmake_options", ["-DCMAKE_PREFIX_PATH=/different/search/path"])):
            with self.subTest(field=field):
                self.assertNotEqual(key, dependencies.cache_key(
                    self.manifest, "caliper", dict(context, **{field: value})))
        for field in ("cc", "cxx", "cmake", "ccache", "system", "machine"):
            changed = dict(context, toolchain=dict(context["toolchain"], **{field: "different"}))
            with self.subTest(toolchain=field):
                self.assertNotEqual(key, dependencies.cache_key(self.manifest, "caliper", changed))
        manifest = copy.deepcopy(self.manifest)
        manifest["dependencies"]["caliper"]["revision"] = "f" * 40
        self.assertNotEqual(key, dependencies.cache_key(manifest, "caliper", context))
        with patch.object(dependencies, "RECIPE_VERSION", dependencies.RECIPE_VERSION + 1):
            self.assertNotEqual(key, dependencies.cache_key(self.manifest, "caliper", context))

    def test_hip_caliper_keys_the_host_compiler_not_hipcc(self):
        context = dict(self.context, backend="hip", cxx="/opt/rocm/bin/hipcc",
                       host_cxx="/usr/bin/g++-12", device_compiler="/opt/rocm/llvm/bin/clang++",
                       architecture="gfx90a", user_cmake_options=[],
                       cmake_options=["-DCMAKE_PREFIX_PATH=/opt/rocm"])
        context["toolchain"] = dict(context["toolchain"], cxx="HIP 6.3.1",
                                    host_cxx="GCC 12.3", device="ROCm Clang")
        key = dependencies.cache_key(self.manifest, "caliper", context)
        changed = dict(context, architecture="gfx942", cxx="/another/hipcc",
                       toolchain=dict(context["toolchain"], cxx="HIP 7", device="new ROCm Clang"))
        self.assertEqual(key, dependencies.cache_key(self.manifest, "caliper", changed))
        changed["toolchain"]["host_cxx"] = "GCC 12.4"
        self.assertNotEqual(key, dependencies.cache_key(self.manifest, "caliper", changed))

    def test_caliper_overrides_keep_conservative_cache_identity(self):
        for explicit_list in (False, True):
            context = dict(self.context, backend="cuda", device_compiler="nvcc",
                           architecture="90", cmake_options=["-DWITH_CUPTI=ON"])
            if explicit_list:
                context["user_cmake_options"] = list(context["cmake_options"])
            with self.subTest(explicit_list=explicit_list):
                self.assertNotEqual(dependencies.cache_key(self.manifest, "caliper", context),
                                    dependencies.cache_key(self.manifest, "caliper",
                                                           dict(context, architecture="100")))

    def test_restored_install_needs_matching_stamp_and_every_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "mfem"
            for relative in dependencies.REQUIRED_FILES["mfem"]:
                file = prefix / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("fixture\n")
            self.assertFalse(dependencies.valid_install(prefix, "mfem", "key"))
            dependencies.write_json(prefix / dependencies.STAMP,
                                    {"key": "key", "prefix": str(prefix.resolve())})
            self.assertTrue(dependencies.valid_install(prefix, "mfem", "key"))
            self.assertFalse(dependencies.valid_install(prefix, "mfem", "another-key"))
            (prefix / "data/fichera-quad.mesh").unlink()
            self.assertFalse(dependencies.valid_install(prefix, "mfem", "key"))

    def test_successful_restore_avoids_build_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            context = dict(self.context, cache_prefix=directory)
            prefix = Path(directory) / "caliper"
            key = dependencies.cache_key(self.manifest, "caliper", context)
            for relative in dependencies.REQUIRED_FILES["caliper"]:
                file = prefix / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("installed\n")
            dependencies.write_json(prefix / dependencies.STAMP,
                                    {"key": key, "prefix": str(prefix.resolve())})
            with patch.object(dependencies, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                rebuilt = dependencies.build_dependency(self.manifest, "caliper", context,
                                                         prefix, Path(directory) / "work", 2)
            self.assertFalse(rebuilt)
            run.assert_not_called()

    def test_failure_keeps_diagnostic_log(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            log = Path(directory) / "failure.log"
            with self.assertRaisesRegex(RuntimeError, "Command failed \\(9\\)"):
                dependencies.run([sys.executable, "-c", "print('intentional diagnostic'); exit(9)"], log)
            self.assertIn("intentional diagnostic", log.read_text())


if __name__ == "__main__":
    unittest.main()
