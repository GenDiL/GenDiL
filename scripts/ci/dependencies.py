#!/usr/bin/env python3
"""Resolve, build, and reuse the third-party libraries used by GenDiL CI."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import ci_process


REPOSITORIES = {
    "mfem": "mfem/mfem",
    "hypre": "hypre-space/hypre",
    "caliper": "llnl/Caliper",
}
STAMP = ".gendil-ci-install.json"
# Bump this if build/install behavior changes without changing recipe options.
RECIPE_VERSION = 1
REQUIRED_FILES = {
    "mfem": [
        "lib/cmake/mfem/MFEMConfig.cmake", "include/mfem.hpp", "lib/libmfem.a",
        "data/star-q3.mesh", "data/pipe-nurbs.mesh",
        "data/square-disc-nurbs.mesh", "data/fichera-quad.mesh",
    ],
    "hypre": [
        "lib/cmake/HYPRE/HYPREConfig.cmake", "include/HYPRE.h", "lib/libHYPRE.a",
        "include/_hypre_utilities.h", "include/_hypre_seq_mv.h",
        "include/_hypre_parcsr_mv.h",
    ],
    "caliper": [
        "share/cmake/caliper/caliper-config.cmake",
        "include/caliper/cali.h", "lib/libcaliper.a",
    ],
}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def github_json(path):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "GenDiL-CI",
               "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request("https://api.github.com/" + path, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        # Do not include request headers or a token in diagnostics.
        raise RuntimeError(f"GitHub API returned HTTP {error.code} for {path}") from None
    except URLError as error:
        raise RuntimeError(f"Could not contact GitHub API: {error.reason}") from None


def resolve_manifest(overrides):
    dependencies = {}
    for name, repository in REPOSITORIES.items():
        ref = overrides.get(name)
        if not ref:
            if name == "mfem":
                ref = github_json(f"repos/{repository}")["default_branch"]
            else:
                release = github_json(f"repos/{repository}/releases/latest")
                if release.get("draft") or release.get("prerelease"):
                    raise RuntimeError(f"The latest {name} release is not stable")
                ref = release["tag_name"]
        # The commits endpoint dereferences both lightweight and annotated tags.
        commit = github_json(f"repos/{repository}/commits/{quote(ref, safe='')}")
        revision = commit["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"GitHub returned an invalid {name} commit SHA")
        dependencies[name] = {"repository": repository, "ref": ref,
                              "revision": revision, "version": ref}
    return {"schema_version": 1,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "dependencies": dependencies}


def dependency(manifest, name):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dependency manifest schema")
    result = manifest["dependencies"][name]
    if result.get("repository") != REPOSITORIES[name]:
        raise ValueError(f"Unexpected repository for {name}")
    if not re.fullmatch(r"[0-9a-f]{40}", result.get("revision", "")):
        raise ValueError(f"The {name} revision must be a resolved, full commit SHA")
    return result


def validate_context(context):
    if context.get("backend") not in ("cpu", "cuda", "hip"):
        raise ValueError("Context backend must be cpu, cuda, or hip")
    for field in ("cc", "cxx", "cache_prefix"):
        if not isinstance(context.get(field), str) or not context[field]:
            raise ValueError(f"Context requires {field}")
    if not Path(context["cache_prefix"]).is_absolute():
        raise ValueError("Context cache_prefix must be absolute")
    if not isinstance(context.get("openmp"), bool):
        raise ValueError("Context openmp must be a boolean")
    if context["backend"] != "cpu":
        if not context.get("device_compiler") or not context.get("architecture"):
            raise ValueError("Device builds require a compiler and explicit architecture")
        if context["architecture"] == "native":
            raise ValueError("CI device architecture cannot depend on GPU discovery")


def cmake_options(name, context, prefix):
    backend = context["backend"]
    cxx = context.get("host_cxx", context["cxx"]) if name == "caliper" else context["cxx"]
    # hipcc otherwise asks for installed GPUs during even a host-only compiler
    # identification probe. Explicit targeting also works on toolkit-only CI.
    cxx_flags = (f"--offload-arch={context['architecture']}"
                 if backend == "hip" and Path(cxx).name == "hipcc" else "")
    options = [
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_INSTALL_LIBDIR=lib", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DBUILD_SHARED_LIBS=OFF", f"-DCMAKE_C_COMPILER={context['cc']}",
        f"-DCMAKE_CXX_COMPILER={cxx}", "-DCMAKE_CXX_STANDARD=20",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON", "-DCMAKE_C_FLAGS=",
        f"-DCMAKE_CXX_FLAGS={cxx_flags}",
        "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG",
    ]
    if context.get("ccache"):
        options.extend(f"-DCMAKE_{language}_COMPILER_LAUNCHER={context['ccache']}"
                       for language in ("C", "CXX"))
    # Caliper's host annotation interface does not need a device compiler.
    if backend != "cpu" and name != "caliper":
        language = backend.upper()
        options += [f"-DCMAKE_{language}_COMPILER={context['device_compiler']}",
                    f"-DCMAKE_{language}_ARCHITECTURES={context['architecture']}",
                    f"-DCMAKE_{language}_STANDARD=20",
                    f"-DCMAKE_{language}_STANDARD_REQUIRED=ON", f"-DCMAKE_{language}_FLAGS=",
                    f"-DCMAKE_{language}_FLAGS_RELEASE=-O3 -DNDEBUG"]
        if context.get("ccache"):
            options.append(f"-DCMAKE_{language}_COMPILER_LAUNCHER={context['ccache']}")
        if backend == "cuda":
            if context.get("cuda_toolkit_root"):
                options.append(f"-DCUDAToolkit_ROOT={context['cuda_toolkit_root']}")
            # Clang compiles CUDA's host and device code itself; only NVCC
            # delegates host compilation to a separate compiler.
            if context.get("device_compiler_id") == "NVIDIA":
                options.append(f"-DCMAKE_CUDA_HOST_COMPILER={context['cxx']}")
        else:
            options += [f"-DHIP_ARCH={context['architecture']}",
                        # hip::device uses this independently of CMake's HIP
                        # language target; omitting it probes local hardware.
                        f"-DAMDGPU_TARGETS={context['architecture']}",
                        f"-DGPU_TARGETS={context['architecture']}",
                        f"-DHIP_HIPCC_FLAGS=--offload-arch={context['architecture']}"]
    openmp = "ON" if context["openmp"] else "OFF"
    if name == "mfem":
        # MFEM has no named modules; automatic scanning adds GCC module flags
        # that prevent ccache from caching ordinary C++ compilation.
        options += ["-DCMAKE_CXX_SCAN_FOR_MODULES=OFF",
                    "-DINSTALL_LIB_DIR=lib", "-DMFEM_PRECISION=double", "-DMFEM_USE_MPI=OFF",
                    "-DMFEM_USE_METIS=OFF", f"-DMFEM_USE_OPENMP={openmp}",
                    "-DMFEM_ENABLE_TESTING=OFF", "-DMFEM_ENABLE_EXAMPLES=OFF",
                    "-DMFEM_ENABLE_MINIAPPS=OFF", "-DMFEM_ENABLE_BENCHMARKS=OFF",
                    "-DMFEM_FETCH_TPLS=OFF", "-DMFEM_USE_RAJA=OFF",
                    "-DMFEM_USE_CALIPER=OFF",
                    f"-DMFEM_USE_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
                    f"-DMFEM_USE_HIP={'ON' if backend == 'hip' else 'OFF'}"]
    elif name == "hypre":
        options += ["-DHYPRE_ENABLE_MPI=OFF", f"-DHYPRE_ENABLE_OPENMP={openmp}",
                    "-DHYPRE_ENABLE_BIGINT=OFF", "-DHYPRE_ENABLE_MIXEDINT=OFF",
                    "-DHYPRE_ENABLE_SINGLE=OFF", "-DHYPRE_ENABLE_LONG_DOUBLE=OFF",
                    "-DHYPRE_ENABLE_COMPLEX=OFF", "-DHYPRE_BUILD_TESTS=OFF",
                    "-DHYPRE_BUILD_EXAMPLES=OFF", "-DHYPRE_ENABLE_FORTRAN=OFF",
                    "-DHYPRE_ENABLE_UMPIRE=OFF", "-DHYPRE_BUILD_UMPIRE=OFF",
                    "-DHYPRE_ENABLE_CALIPER=OFF", "-DHYPRE_ENABLE_SYCL=OFF",
                    f"-DHYPRE_ENABLE_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
                    f"-DHYPRE_ENABLE_HIP={'ON' if backend == 'hip' else 'OFF'}"]
    else:
        options += ["-DBUILD_TESTING=OFF", "-DBUILD_DOCS=OFF", "-DWITH_TOOLS=OFF",
                    "-DWITH_FORTRAN=OFF", "-DWITH_MPI=OFF", "-DWITH_GOTCHA=OFF",
                    "-DWITH_KOKKOS=OFF", "-DWITH_CUPTI=OFF", "-DWITH_NVTX=OFF",
                    "-DWITH_ROCPROFILER=OFF", "-DWITH_ROCM=OFF",
                    "-DINSTALL_CONFIG=ON", "-DINSTALL_HEADERS=ON"]
    options += context.get("cmake_options", [])
    return options


def cache_context(name, context):
    # Overrides can enable device services or replace compilers. Keep their
    # identity conservative, including legacy contexts without a separate list.
    overrides = context.get("user_cmake_options", context.get("cmake_options", []))
    if name != "caliper" or overrides:
        return context
    toolchain = context.get("toolchain", {})
    cxx_version = "host_cxx" if "host_cxx" in context else "cxx"
    return {
        "container": context.get("container"), "generator": context.get("generator"),
        "cache_prefix": context["cache_prefix"], "cc": context["cc"],
        "cxx": context.get("host_cxx", context["cxx"]), "ccache": context.get("ccache"),
        "toolchain": {**{key: toolchain.get(key) for key in
                         ("cc", "cmake", "ccache", "system", "machine")},
                      "cxx": toolchain.get(cxx_version)},
    }


def cache_key(manifest, name, context):
    validate_context(context)
    spec = dependency(manifest, name)
    # Other libraries' revisions and resolution timestamps intentionally do not
    # enter this key. Updating MFEM should not rebuild unchanged Hypre/Caliper.
    identity = {
        "recipe_version": RECIPE_VERSION, "name": name,
        "repository": spec["repository"], "revision": spec["revision"],
        "context": cache_context(name, context),
        "options": cmake_options(name, context, str(Path(context["cache_prefix"]) / name)),
        "required_files": REQUIRED_FILES[name],
        "targets": ["mfem", "copy_data"] if name == "mfem" else [],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def valid_install(prefix, name, key):
    prefix = Path(prefix)
    try:
        stamp = read_json(prefix / STAMP)
    except (OSError, ValueError):
        return False
    return (stamp.get("key") == key and stamp.get("prefix") == str(prefix.resolve())
            and all((prefix / relative).is_file() and (prefix / relative).stat().st_size
                    for relative in REQUIRED_FILES[name]))


def run(command, log, *, cwd=None):
    ci_process.run(command, log, cwd=cwd, append=True)


def build_dependency(manifest, name, context, prefix, work_dir, jobs):
    key = cache_key(manifest, name, context)
    spec = dependency(manifest, name)
    prefix = Path(prefix).resolve()
    expected_prefix = (Path(context["cache_prefix"]) / name).resolve()
    if prefix != expected_prefix:
        raise ValueError(f"Install prefix must match context.cache_prefix/name: {expected_prefix}")
    if valid_install(prefix, name, key):
        print(f"Reusing verified {name} installation: {prefix}")
        return False

    work = Path(work_dir).resolve() / name
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source"
    binary = work / "build"
    # These paths belong to this helper. Never install over an older partial
    # prefix, because stale exports or libraries can conceal a failed upgrade.
    if prefix.exists():
        shutil.rmtree(prefix)
    if binary.exists():
        shutil.rmtree(binary)
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    print(f"Building {name} {spec['revision']} for {context['backend']}", flush=True)
    fetch_log = work / "fetch.log"
    run(["git", "init", "--quiet", str(source)], fetch_log)
    run(["git", "remote", "add", "origin", f"https://github.com/{spec['repository']}.git"],
        fetch_log, cwd=source)
    run(["git", "fetch", "--depth=1", "origin", spec["revision"]], fetch_log, cwd=source)
    run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], fetch_log, cwd=source)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != spec["revision"]:
        raise RuntimeError(f"Fetched {name} revision does not match the manifest")

    source_dir = source / "src" if name == "hypre" else source
    configure = ["cmake", "-S", str(source_dir), "-B", str(binary),
                 "-G", context.get("generator", "Ninja")]
    configure += cmake_options(name, context, prefix)
    run(configure, work / "configure.log")
    command = ["cmake", "--build", str(binary), "--parallel", str(jobs)]
    if name == "mfem":
        command += ["--target", "mfem", "copy_data"]
    run(command, work / "build.log")
    run(["cmake", "--install", str(binary)], work / "install.log")
    missing = [path for path in REQUIRED_FILES[name] if not (prefix / path).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete {name} installation: {', '.join(missing)}")
    write_json(prefix / STAMP, {"key": key, "prefix": str(prefix),
                               "dependency": spec, "context": context})
    if not valid_install(prefix, name, key):
        raise RuntimeError(f"The {name} installation failed post-install validation")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve", help="Resolve dependency revisions once")
    resolve.add_argument("--output", required=True, type=Path)
    for name in REPOSITORIES:
        resolve.add_argument(f"--{name}-ref")
    for action in ("key", "validate", "build"):
        command = commands.add_parser(action)
        command.add_argument("--manifest", required=True, type=Path)
        command.add_argument("--context", required=True, type=Path)
        command.add_argument("--name", required=True, choices=REPOSITORIES)
        if action != "key":
            command.add_argument("--prefix", required=True, type=Path)
        if action == "build":
            command.add_argument("--work-dir", required=True, type=Path)
            command.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "resolve":
        manifest = resolve_manifest({name: getattr(args, f"{name}_ref") for name in REPOSITORIES})
        write_json(args.output, manifest)
        for name, spec in manifest["dependencies"].items():
            print(f"{name}: {spec['ref']} -> {spec['revision']}")
        return 0
    manifest, context = read_json(args.manifest), read_json(args.context)
    key = cache_key(manifest, args.name, context)
    if args.command == "key":
        print(key)
    elif args.command == "validate":
        return 0 if valid_install(args.prefix, args.name, key) else 1
    else:
        if args.jobs < 1:
            raise ValueError("--jobs must be positive")
        build_dependency(manifest, args.name, context, args.prefix, args.work_dir, args.jobs)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Dependency setup failed: {error}", file=sys.stderr)
        sys.exit(1)
