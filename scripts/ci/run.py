#!/usr/bin/env python3
"""Reproduce a CI build using a named configuration from configurations.json.

Run prepare, dependencies, check, then report; report also collects diagnostics
after a failed phase. Preparation needs an installed toolchain and, for optional
libraries, a manifest from dependencies.py resolve. minimal needs no manifest.
matrix prints job selections without preparing tools. GitHub Actions handles
scheduling and cache/artifact transfers. See README.md for the full workflow,
overrides, output locations, and GPU compilation limits.
"""

# Minimal example, from the repository root with the toolchain already installed:
#   python3 scripts/ci/run.py prepare --configuration minimal
#   python3 scripts/ci/run.py dependencies --configuration minimal
#   python3 scripts/ci/run.py check --configuration minimal
#   python3 scripts/ci/run.py report --configuration minimal

import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import dependencies
import ci_coverage as coverage
import ci_process

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path(__file__).with_name("configurations.json")
CONFIG = json.loads(CONFIG_FILE.read_text())
RELEASE_FLAGS = "-O3 -DNDEBUG"
SAFE_TEST_LABELS = coverage.SAFE_TEST_LABELS
PHASES = ("prepare", "hypre", "caliper", "mfem", "configure", "build", "test", "analysis")
# Source diagnostics and their report copies must both be invalidated on retries.
CONFIGURE_DIAGNOSTICS = ("CMakeCache.txt", "compile_commands.json", "GENDILConfig.cmake",
                         "CMakeFiles/CMakeConfigureLog.yaml")
TEST_DIAGNOSTICS = ("Testing/Temporary/LastTest.log", "Testing/Temporary/LastTestsFailed.log")
CHECK_REPORTS = ("configure.log", "build.log", "ctest.log", "clang-tidy.log", "junit.xml",
                 "coverage.json", "coverage.txt", "features.txt", "excluded-tests.txt",
                 "ctest-inventory.json", "ctest-inventory.log",
                 "ctest-gpu-selection.json", "ctest-gpu-selection.log",
                 "installed-consumer", "configure-probes") + tuple(
                     Path(path).name for path in CONFIGURE_DIAGNOSTICS + TEST_DIAGNOSTICS)


def configuration(name):
    return CONFIG["configurations"][name]


def paths(args):
    """Return the work root, configuration build directory, and report directory."""
    work = Path(args.work_dir).resolve()
    return work, work / "build" / args.configuration, work / "reports" / args.configuration


def dependency_prefix(work, profile):
    """Choose a stable install prefix outside the source tree for CMake exports."""
    base = work / "dependencies"
    if base.is_relative_to(ROOT):
        temporary = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()).resolve()
        base = temporary / "gendil-ci-dependencies" / digest(str(ROOT))[:12]
    if base.is_relative_to(ROOT):
        raise RuntimeError("Set --work-dir or RUNNER_TEMP outside the GenDiL source tree for dependency installations")
    return base / profile


def execute(command, log, env=None):
    """Retain complete diagnostics and propagate the original failure."""
    ci_process.run(command, log, env=env)


def phase_status(reports):
    path = reports / "status.json"
    return json.loads(path.read_text()) if path.exists() else dict.fromkeys(PHASES, "unattempted")


def set_status(reports, **states):
    reports.mkdir(parents=True, exist_ok=True)
    status = phase_status(reports)
    status.update(states)
    (reports / "status.json").write_text(json.dumps(status, indent=2) + "\n")


@contextmanager
def phase(reports, name):
    """Record running/passed/failed for this attempt, preserving raised exceptions."""
    set_status(reports, **{name: "running"})
    try:
        yield
    except BaseException:
        set_status(reports, **{name: "failed"})
        raise
    else:
        set_status(reports, **{name: "passed"})


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def reset_check(args):
    """Invalidate downstream results while retaining compiled objects and caches.

    Removing CMakeCache.txt refreshes capability checks during reconfiguration.
    Removing raw diagnostic files also prevents report() from copying old results
    back out of a reused build tree after an early failure.
    """
    _, build, reports = paths(args)
    for name in CHECK_REPORTS:
        remove(reports / name)
    set_status(reports, **dict.fromkeys(("configure", "build", "test", "analysis"), "unattempted"))
    for relative in CONFIGURE_DIAGNOSTICS + TEST_DIAGNOSTICS:
        remove(build / relative)
    for name in ("configure-tests", "installed-package-test"):
        for source in diagnostic_files(build / name):
            source.unlink()


def diagnostic_files(directory):
    """Select logs and configuration diagnostics, leaving build artifacts alone."""
    return (path for path in directory.rglob("*") if path.is_file() and
            (path.suffix == ".log" or path.name in ("CMakeCache.txt", "CMakeConfigureLog.yaml")))


def package_inventory(reports):
    """Capture installed versions for diagnosis; they are not cache inputs."""
    inventory = {"python": sys.version, "python_build_tools": {}}
    for name in ("cmake", "ninja", "pip", "setuptools", "wheel"):
        try:
            inventory["python_build_tools"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    if shutil.which("dpkg-query"):
        try:
            output = subprocess.check_output(
                ["dpkg-query", "--show", "--showformat=${binary:Package}\t${Version}\t${Architecture}\t${db:Status-Status}\n"],
                text=True, stderr=subprocess.STDOUT)
            inventory["apt_packages"] = sorted(
                [fields[:3] for line in output.splitlines()
                 if len(fields := line.split("\t")) == 4 and fields[3] == "installed"])
        except (OSError, subprocess.CalledProcessError) as error:
            inventory["apt_unavailable"] = str(error)
    else:
        inventory["apt_unavailable"] = "dpkg-query is not available on this platform"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "packages.json").write_text(json.dumps(inventory, indent=2) + "\n")


def tool(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Required tool not found: {name}")
    # Preserve compiler symlinks: clang++ and clang select different link modes.
    return os.path.abspath(found)


def version(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cpu_fingerprint(cpuinfo):
    # Frequencies, processor numbering, and APIC IDs vary between identical VMs.
    fields = {"vendor_id", "cpu family", "model", "model name", "stepping", "flags",
              "CPU implementer", "CPU architecture", "CPU variant", "CPU part",
              "CPU revision", "Features", "Hardware"}
    return sorted({line.strip() for line in cpuinfo.splitlines()
                   if line.partition(":")[0].strip() in fields})


def prepare(args):
    """Start a fresh reporting attempt before discovering the installed tools."""
    _, _, reports = paths(args)
    remove(reports)
    with phase(reports, "prepare"):
        package_inventory(reports)
        prepare_context(args)


def prepare_context(args):
    """Save effective tools/options, dependency revisions, and cache identities.

    Later phases read this snapshot. Optional-library configurations require a
    resolved manifest; dependency-free configurations record an empty set instead.
    --github-output additionally exposes cache keys and prefixes to the workflow.
    """
    cfg = configuration(args.configuration)
    work, _, reports = paths(args)
    reports.mkdir(parents=True, exist_ok=True)
    cc = tool(os.environ.get("CC", cfg["cc"]))
    cxx = tool(os.environ.get("CXX", cfg["cxx"]))
    device = tool(cfg["device_compiler"]) if cfg.get("device_compiler") else None
    context = {
        "backend": cfg["backend"], "cc": cc, "cxx": cxx,
        "device_compiler": device, "architecture": cfg.get("architecture"),
        "openmp": cfg["openmp"], "generator": os.environ.get("CMAKE_GENERATOR", "Ninja"),
        "container": os.environ.get("GENDIL_CI_IMAGE", "local"),
        "ccache": shutil.which("ccache"),
        "cache_prefix": str(dependency_prefix(work, cfg["profile"])),
        "user_cmake_options": shlex.split(os.environ.get("GENDIL_CI_CMAKE_OPTIONS", "")),
        "toolchain": {
            "cc": version([cc, "--version"]), "cxx": version([cxx, "--version"]),
            "cmake": version(["cmake", "--version"]),
            "system": platform.system(), "machine": platform.machine(),
        },
    }
    context["cmake_options"] = list(context["user_cmake_options"])
    if device:
        context["toolchain"]["device"] = version([device, "--version"])
    if cfg["backend"] == "cuda":
        context["device_compiler_id"] = cfg["device_compiler_id"]
        context["cuda_toolkit_root"] = cfg["cuda_toolkit_root"]
        context["toolchain"]["cuda_toolkit"] = version(
            [str(Path(cfg["cuda_toolkit_root"]) / "bin/nvcc"), "--version"])
    if context["ccache"]:
        context["toolchain"]["ccache"] = version([context["ccache"], "--version"])
    if cfg["backend"] == "hip":
        # Keep the default overridable by explicit, including typed, CMake options.
        context["cmake_options"].insert(0, "-DCMAKE_PREFIX_PATH=/opt/rocm")
        # hipcc is needed by MFEM's FindHIP integration; nested CPU probes use g++.
        context["host_cxx"] = tool(os.environ.get("GENDIL_CI_HOST_CXX", cfg["host_cxx"]))
        context["toolchain"]["host_cxx"] = version([context["host_cxx"], "--version"])
    (reports / "context.json").write_text(json.dumps(context, indent=2) + "\n")
    manifest = (json.loads(Path(args.manifest).read_text()) if cfg["tpls"]
                else {"schema_version": 1, "dependencies": {}})
    (reports / "dependencies.json").write_text(json.dumps(manifest, indent=2) + "\n")
    keys = {}
    if cfg["tpls"]:
        for name in ("mfem", "hypre", "caliper"):
            command = [sys.executable, str(Path(__file__).with_name("dependencies.py")), "key",
                       "--manifest", str(Path(args.manifest).resolve()), "--name", name,
                       "--context", str(reports / "context.json")]
            keys[name + "_key"] = subprocess.check_output(command, text=True).strip()
    # Include driver changes and native CPU features in compiler-cache isolation.
    # Dependency installation keys are computed separately by dependencies.py.
    cache_identity = {"context": context, "configuration": cfg,
                      "driver": Path(__file__).read_text()}
    if cfg.get("native"):
        cpu_info = Path("/proc/cpuinfo")
        cache_identity["cpu"] = cpu_fingerprint(cpu_info.read_text()) if cpu_info.exists() else platform.processor()
    outputs = {"uses_tpls": str(cfg["tpls"]).lower(),
               "dependency_prefix": context["cache_prefix"],
               "ccache_key": digest(cache_identity), **keys}
    (reports / "cache-keys.json").write_text(json.dumps(outputs, indent=2) + "\n")
    build_manifest = {**manifest, "configuration": args.configuration,
                      "cache_namespace": os.environ.get("GENDIL_CI_CACHE_NAMESPACE", "v1"),
                      "settings": cfg, "context": context, "cache_keys": outputs,
                      "dependency_build_options": {
                          name: dependencies.cmake_options(
                              name, context, str(Path(context["cache_prefix"]) / name))
                          for name in ("hypre", "caliper", "mfem") if cfg["tpls"]}}
    (reports / "manifest.json").write_text(json.dumps(build_manifest, indent=2) + "\n")
    if args.github_output:
        with open(args.github_output, "a") as stream:
            for key, value in outputs.items():
                stream.write(f"{key}={value}\n")
    print(json.dumps({"configuration": args.configuration, **outputs}, indent=2))


def context_for(args):
    """Load saved settings; rerun prepare to apply changed environment overrides."""
    _, _, reports = paths(args)
    return json.loads((reports / "context.json").read_text())


def dependency_build(args):
    """Validate/build prepared dependencies and invalidate downstream check reports.

    The default order builds Hypre before MFEM. --dependency selects one library
    so the workflow can save each installation before starting the next build.
    """
    cfg = configuration(args.configuration)
    if not cfg["tpls"]:
        return
    work, _, reports = paths(args)
    reset_check(args)
    context = context_for(args)
    for name in ([args.dependency] if args.dependency else ("hypre", "caliper", "mfem")):
        remove(reports / "dependency-logs" / name)
        dependency_work = work / "dependency-build" / cfg["profile"] / name
        for filename in ("fetch.log", "configure.log", "build.log", "install.log"):
            remove(dependency_work / filename)
        for source in diagnostic_files(dependency_work / "build"):
            source.unlink()
        with phase(reports, name):
            execute([sys.executable, "-u", str(Path(__file__).with_name("dependencies.py")), "build",
                     "--manifest", str(reports / "dependencies.json"), "--name", name,
                     "--context", str(reports / "context.json"),
                     "--prefix", str(Path(context["cache_prefix"]) / name),
                     "--work-dir", str(work / "dependency-build" / cfg["profile"]),
                     "--jobs", str(args.jobs)], reports / f"dependency-{name}.log")


def find_package_dir(prefix, filename):
    files = sorted(prefix.rglob(filename))
    if len(files) != 1:
        raise RuntimeError(f"Expected one {filename} beneath {prefix}, found {len(files)}")
    return str(files[0].parent)


def configure_command(args, context):
    """Construct GenDiL's CMake arguments from the prepared tools and configuration.

    Explicit device targets avoid hardware discovery on CPU runners. Saved CMake
    options come last, allowing local overrides of the generated defaults.
    """
    cfg = configuration(args.configuration)
    _, build, _ = paths(args)
    flags = RELEASE_FLAGS + (" -march=native -funroll-loops" if cfg.get("native") else "")
    cmd = ["cmake", "-S", str(ROOT), "-B", str(build), "-G", context["generator"],
           "-DCMAKE_BUILD_TYPE=" + ("Debug" if args.configuration == "sanitizers" else "Release"),
           "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DBUILD_TESTING=ON",
           f"-DGENDIL_ENABLE_BENCHMARKS={'ON' if cfg['benchmarks'] else 'OFF'}", "-DUSE_RAJA=OFF",
           f"-DCMAKE_C_COMPILER={context['cc']}", f"-DCMAKE_CXX_COMPILER={context['cxx']}",
           f"-DCMAKE_C_FLAGS_RELEASE={RELEASE_FLAGS}", f"-DCMAKE_CXX_FLAGS_RELEASE={flags}",
           f"-DUSE_OPENMP={'ON' if cfg['openmp'] else 'OFF'}",
           f"-DUSE_CUDA={'ON' if cfg['backend'] == 'cuda' else 'OFF'}",
           f"-DUSE_HIP={'ON' if cfg['backend'] == 'hip' else 'OFF'}",
           "-DGENDIL_FETCH_ROCPRIM=OFF",
           "-DGENDIL_DEVICE_SPARSE_FINALIZATION=" + ("OFF" if cfg["backend"] == "cpu" else "ON")]
    for name in ("MFEM", "HYPRE", "CALIPER"):
        cmd.append(f"-DUSE_{name}={'ON' if cfg['tpls'] else 'OFF'}")
    if cfg["tpls"]:
        prefix = Path(context["cache_prefix"])
        for dependency, package, variable in (("mfem", "MFEMConfig.cmake", "MFEM_DIR"),
                ("hypre", "HYPREConfig.cmake", "HYPRE_DIR"),
                ("caliper", "caliper-config.cmake", "caliper_DIR")):
            cmd.append(f"-D{variable}={find_package_dir(prefix / dependency, package)}")
    if cfg["backend"] != "cpu":
        language = cfg["backend"].upper()
        cmd += [f"-DCMAKE_{language}_COMPILER={context['device_compiler']}",
                f"-DCMAKE_{language}_ARCHITECTURES={context['architecture']}",
                f"-DCMAKE_{language}_FLAGS_RELEASE={RELEASE_FLAGS}"]
        if cfg["backend"] == "cuda":
            cmd.append(f"-DCUDAToolkit_ROOT={context['cuda_toolkit_root']}")
            if context["device_compiler_id"] == "NVIDIA":
                cmd.append(f"-DCMAKE_CUDA_HOST_COMPILER={context['cxx']}")
        else:
            cmd += ["-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm",
                    f"-DAMDGPU_TARGETS={context['architecture']}",
                    f"-DGENDIL_TEST_HOST_C_COMPILER={context['cc']}",
                    f"-DGENDIL_TEST_HOST_CXX_COMPILER={context['host_cxx']}"]
            if Path(context["cxx"]).name == "hipcc":
                cmd.append(f"-DCMAKE_CXX_FLAGS=--offload-arch={context['architecture']}")
    if context["ccache"]:
        for language in ("C", "CXX", "CUDA", "HIP"):
            cmd.append(f"-DCMAKE_{language}_COMPILER_LAUNCHER={context['ccache']}")
    if args.configuration == "sanitizers":
        sanitizers = "-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
        cmd += [f"-DCMAKE_C_FLAGS_DEBUG={sanitizers}", f"-DCMAKE_CXX_FLAGS_DEBUG={sanitizers}",
                "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined"]
    return cmd + context["cmake_options"]


def check_features(build, cfg, reports):
    """Report selected features and require device Hypre/finalization in GPU jobs."""
    cache = (build / "CMakeCache.txt").read_text()
    selected = [line for line in cache.splitlines() if re.match(
        r"(?:USE_|GENDIL_|OpenMP_|CMAKE_.*(?:COMPILER|ARCHITECTURES|FLAGS))", line)]
    config = (build / "GENDILConfig.cmake").read_text()
    capabilities = []
    commands = build / "compile_commands.json"
    if commands.is_file():
        definitions = commands.read_text()
        vendor_features = {
            "cuda": ("GENDIL_CUSPARSE_HAS_GENERIC_BSR", "GENDIL_CUSPARSE_HAS_SPMV_PREPROCESS",
                     "GENDIL_CUSPARSE_HAS_FLOAT_DOUBLE_SPMV"),
            "hip": ("GENDIL_ROCSPARSE_HAS_GENERIC_BSR",),
        }
        for feature in vendor_features.get(cfg["backend"], ()):
            enabled = re.search(r"-D" + feature + r"\b", definitions) is not None
            capabilities.append(f"{feature}: {'ON' if enabled else 'OFF'}")
    (reports / "features.txt").write_text("\n".join(capabilities + selected) + "\n" + config)
    if cfg["backend"] != "cpu":
        for feature in ("GENDIL_USE_HYPRE_DEVICE", "GENDIL_HAS_DEVICE_SPARSE_FINALIZATION"):
            if not re.search(r"set\(\s*" + feature + r'\s+"?ON"?\s*\)', config):
                raise RuntimeError(f"GPU CI requires {feature}=ON; see features.txt")


def check(args):
    """Configure, then run analysis or build and execute the configuration's CTest scope.

    CPU test configurations run all registered tests. GPU configurations compile
    their default targets and run labeled host checks, including compilation of
    the installed consumer; they do not execute GPU programs.
    """
    cfg = configuration(args.configuration)
    _, build, reports = paths(args)
    reset_check(args)
    context = context_for(args)
    coverage.request(build)
    with phase(reports, "configure"):
        execute(configure_command(args, context), reports / "configure.log")
        coverage.collect(build, reports, cfg)
        check_features(build, cfg, reports)
    configure_log = (reports / "configure.log").read_text()
    (reports / "excluded-tests.txt").write_text("\n".join(
        line for line in configure_log.splitlines() if "Skipping test " in line) + "\n")
    if args.configuration == "analysis":
        with phase(reports, "analysis"):
            executable = tool(os.environ.get("GENDIL_CLANG_TIDY_RUNNER", "run-clang-tidy-18"))
            execute([executable, "-p", str(build), "-j", str(args.jobs),
                     "-clang-tidy-binary", tool(os.environ.get("GENDIL_CLANG_TIDY", "clang-tidy-18"))],
                    reports / "clang-tidy.log")
        return
    with phase(reports, "build"):
        execute(["cmake", "--build", str(build), "--parallel", str(args.jobs)], reports / "build.log")
    command = ["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error",
               "--timeout", "300", "--parallel", "1", "--output-junit", str(reports / "junit.xml")]
    if cfg["backend"] != "cpu":
        command += ["-L", SAFE_TEST_LABELS]
    environment = dict(os.environ, OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE")
    if args.configuration == "sanitizers":
        environment.update(ASAN_OPTIONS="detect_leaks=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    with phase(reports, "test"):
        execute(command, reports / "ctest.log", env=environment)


def report(args):
    """Collect diagnostics only for phases attempted since the latest prepare.

    This is safe to repeat after a failure: unattempted phases do not contribute
    logs from older build trees. Missing optional inventory tools remain nonfatal.
    """
    work, build, reports = paths(args)
    reports.mkdir(parents=True, exist_ok=True)
    status = phase_status(reports)
    if not (reports / "packages.json").exists():
        package_inventory(reports)
    files = CONFIGURE_DIAGNOSTICS if status["configure"] != "unattempted" else ()
    if status["test"] != "unattempted":
        files += TEST_DIAGNOSTICS
    for relative in files:
        source = build / relative
        if source.is_file():
            shutil.copy2(source, reports / source.name)
    directories = []
    if status["test"] != "unattempted":
        directories += [(build / "installed-package-test", "installed-consumer"),
                        (build / "configure-tests", "configure-probes")]
    for name in ("hypre", "caliper", "mfem"):
        if status[name] != "unattempted":
            directory = work / "dependency-build" / configuration(args.configuration)["profile"] / name
            # Source trees may contain unrelated log fixtures. Only collect build diagnostics.
            directories.append((directory / "build", f"dependency-logs/{name}/build"))
            for filename in ("fetch.log", "configure.log", "build.log", "install.log"):
                source = directory / filename
                if source.is_file():
                    target = reports / "dependency-logs" / name / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    for directory, destination in directories:
        for source in diagnostic_files(directory):
            target = reports / destination / source.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if shutil.which("ccache"):
        try:
            execute(["ccache", "--show-stats"], reports / "ccache.txt")
        except (OSError, RuntimeError) as error:
            print(f"Compiler-cache statistics unavailable: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("matrix", "prepare", "dependencies", "check", "report"))
    parser.add_argument("--configuration", choices=CONFIG["configurations"], default="minimal")
    parser.add_argument("--work-dir", default=str(ROOT / ".ci"))
    parser.add_argument("--manifest", default=str(ROOT / ".ci" / "dependencies.json"))
    parser.add_argument("--github-output")
    parser.add_argument("--include-sanitizers", action="store_true")
    parser.add_argument("--dependency-free", action="store_true",
                        help="Limit the matrix to configurations without optional libraries")
    parser.add_argument("--dependency", choices=("hypre", "caliper", "mfem"),
                        help="Build/validate only this dependency in the dependencies step")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.command == "matrix":
        entries = [{"configuration": name, "image": CONFIG["images"][cfg.get("image", cfg["backend"])], "tpls": cfg["tpls"],
                    "timeout": 90 if cfg["backend"] == "cpu" else 180}
                   for name, cfg in CONFIG["configurations"].items()
                   if (name != "sanitizers" or args.include_sanitizers)
                   and (not args.dependency_free or not cfg["tpls"])]
        print(json.dumps({"include": entries}, separators=(",", ":")))
        return
    if args.command == "prepare":
        prepare(args)
    elif args.command == "dependencies":
        dependency_build(args)
    elif args.command == "check":
        check(args)
    else:
        report(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"CI error: {error}", file=sys.stderr)
        sys.exit(1)
