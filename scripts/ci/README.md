# Reproduce a CI build

For the jobs' purpose and how the workflow fits together, start with the
[CI workflow overview](../../.github/workflows/README.md).

These helpers build GenDiL with the same dependencies and options used by GitHub
Actions. CPU test jobs run the complete test suite; the analysis job runs
clang-tidy. CUDA and HIP jobs compile device code and run configuration/package
checks on ordinary CPU runners; they do not execute GPU kernels or validate
numerical GPU results.

Run these commands from the repository root, with the selected compiler,
CMake 3.30.5, Ninja, Python 3.10 or newer, and optionally ccache available:

```bash
python3 scripts/ci/run.py prepare --configuration minimal
python3 scripts/ci/run.py dependencies --configuration minimal
python3 scripts/ci/run.py check --configuration minimal
python3 scripts/ci/run.py report --configuration minimal
```

The `dependencies` step does nothing for `minimal`; no dependency manifest or
network resolution is needed. For a configuration with optional libraries, first
run `python3 scripts/ci/dependencies.py resolve --output .ci/dependencies.json`,
then replace `minimal` in the commands above with that configuration.
`--jobs` controls build
parallelism and defaults to two. Hosted analysis uses four workers to complete
the full clang-tidy pass within its timeout; other jobs use two.
`--work-dir` and `--manifest` select different
build/cache and dependency-manifest locations.
`check` refreshes the CMake configuration so capability probes reflect the selected
dependencies, including when reusing a local work directory.

| Configuration | Dependencies and execution |
| --- | --- |
| `gcc` | GCC 14, OpenMP, MFEM/Hypre/Caliper; full CPU tests |
| `clang` | Clang 18, OpenMP, MFEM/Hypre/Caliper; full CPU tests |
| `minimal` | GCC 14, no optional TPLs or OpenMP; full CPU tests |
| `cuda-nvcc` | NVCC 12.4.1/GCC 12, architecture 90, matching MFEM/Hypre, Caliper; compilation and host-only checks |
| `cuda-clang` | Clang 19/CUDA 12.4.1, architecture 90, matching MFEM/Hypre, Caliper; compilation and host-only checks |
| `hip` | ROCm 6.3.1, architecture gfx90a, matching MFEM/Hypre, Caliper; compilation and host-only checks |
| `analysis` | Clang 18 with all supported TPLs; full clang-tidy analysis |
| `sanitizers` | Clang 18, no optional TPLs/OpenMP; ASan and UBSan |

RAJA and GPU profiling services are not enabled. MPI is disabled. The first seven
configurations run on PRs, pushes to `main`, weekly schedules, and manual runs;
sanitizers run weekly and on manual runs. Every Release compiler job builds all
registered tests, examples, and benchmarks in the default build. Examples and
benchmarks are compiled and linked, but not executed. The analysis job includes
benchmark sources; the sanitizer job leaves benchmarks disabled. Performance
measurements and thresholds remain outside this workflow.
Dependency discovery failures leave the workflow failed, but still allow
`minimal` and any scheduled/manual sanitizer configuration to run. These jobs
wait for discovery to finish. Helper-test failures prevent all builds.

## CUDA compiler and target coverage

CUDA has two independent jobs because NVCC and Clang accept different subsets of
the test suite. NVCC excludes weak-form DSL tests affected by its template
compilation bugs, as well as the other explicitly excluded tests. These tests
remain enabled for Clang CUDA, HIP, and CPU builds. NVCC still compiles the
registered low-level arithmetic, geometry, device, and legacy-operator tests.
Clang 19 follows the compiler family in the repository's
Matrix machine configuration. Both jobs use the same CUDA toolkit and GPU
architecture, and each builds its own CUDA-enabled MFEM/Hypre installations.
The installed consumer uses the same CUDA compiler as its producer.
CUDA 12.4.1 replaces 12.2.2 because the older NVCC/GCC 12 combination fails on
the C++20 `std::variant` used by current MFEM; NVIDIA reports the reproducer
working with [CUDA 12.4.1](https://forums.developer.nvidia.com/t/nvcc-c-20-std-variant-complie-failed/270162/5).

Configured executable counts and excluded test names are recorded in each job's
coverage reports; they are not counts of executed GPU tests. The
filtered-synchronization experiment remains excluded from the default
build. Six 4D–6D adaptive-advection benchmarks retain their pre-existing exclusion
for **both** CUDA compilers; CPU/HIP builds register 54 benchmarks. These exclusions
need separate compilation evidence before they can be narrowed to NVCC alone.

Each job records the actual configured target inventory and CTest selection in
`coverage.json` and `coverage.txt`. This makes compiler-specific omissions visible
without treating a reduced NVCC build as complete CUDA coverage. The report checks
the selected CUDA compiler identity, the presence of tests/examples/benchmarks,
and device-language compilation of their sources. The target inventory describes
configuration; a successful build log establishes compilation, and JUnit records
only the tests actually executed.

## Toolchains and local overrides

`configurations.json` records the compilers, GPU architectures, and immutable
container images. The workflow uses standard Ubuntu CPU runners and executes
each configuration inside its selected container. `bootstrap.sh CONFIGURATION`
installs tools **inside that Ubuntu container as root**; it is not a workstation
installer. When running it outside GitHub Actions, add
`/opt/gendil-ci-tools/bin` to `PATH` afterward.
The Clang CUDA job installs Clang 19 and its OpenMP runtime from the
[LLVM Jammy repository](https://apt.llvm.org/), with its signing-key fingerprint
checked and the key restricted to that repository.

For an existing local toolchain, set `CC` and `CXX` before `prepare`. Set
`CMAKE_GENERATOR` if Ninja is unavailable. Additional local CMake arguments can
be supplied through `GENDIL_CI_CMAKE_OPTIONS`, using shell-style quoting. These
settings are recorded in the context and dependency cache keys. For example:

```bash
CC=clang CXX=clang++ CMAKE_GENERATOR='Unix Makefiles' \
  python3 scripts/ci/run.py prepare --configuration minimal
python3 scripts/ci/run.py check --configuration minimal
```

All Release configurations explicitly use `-O3 -DNDEBUG`. The `gcc` GenDiL build
also enables `-march=native -funroll-loops`; its compiler-cache key includes stable
CPU identity/features. Dependencies use portable CPU flags. Sanitizers use
explicit instrumentation flags in a separate Debug configuration.

## Reproduce dependency revisions

Each workflow resolves MFEM's current development commit and the latest stable
Hypre and Caliper releases once. Jobs using optional libraries share that manifest. A tag is
resolved to its full commit SHA before building. No compatibility failure causes
an automatic downgrade or disables a required interface.

Download `dependency-manifest` from a failed run and use it without resolving
again, or request specific refs:

```bash
python3 scripts/ci/dependencies.py resolve --output .ci/dependencies.json \
  --mfem-ref COMMIT --hypre-ref TAG_OR_COMMIT --caliper-ref TAG_OR_COMMIT
```

The same three overrides are available under **Run workflow** on GitHub. Upgrade
toolkits deliberately when a new dependency requires them. The feature report
records the optional cuSPARSE capabilities selected by each toolchain.
The former `cuda` configuration is now named `cuda-nvcc`; use `cuda-clang` to
reproduce the other CUDA frontend.

## Caches and diagnostics

MFEM, Hypre, and Caliper have separate installation caches. Their keys include
the library revision, toolchain/container, architecture, build recipe/options,
and installation path. Only an exact key and a valid installation stamp permit
reuse. Required headers, libraries, package files, and MFEM mesh fixtures are
checked after restoration. Updating MFEM does not invalidate unchanged Hypre or
Caliper caches. MFEM's `copy_data` target is built so cached installs retain the
meshes used by integration tests.
Dependency installations must live outside the GenDiL source tree so CMake can
export their include paths. With the default work directory they use a stable
repository-specific directory under `RUNNER_TEMP` (or the local temporary
directory). An external `--work-dir` keeps them under that directory instead.
The manifest records the exact prefixes; build objects and reports remain in
the selected work directory.
NVCC and Clang CUDA use separate dependency prefixes and cache identities;
compiler and toolkit versions are included in the manifest.
Caliper's default host-only cache uses its actual host toolchain and recipe;
unused GPU compiler and architecture changes do not invalidate it. User CMake
overrides retain conservative cache identities.

An incomplete restored installation is rebuilt. GitHub cache entries are
immutable, so delete a damaged entry in Actions if it is repeatedly restored;
the next successful run will save a fresh installation under the same key.

Compiler caches are bounded to 1 GiB per job and can reuse compilation work across
commits. Complete build trees and NVCC intermediates are not cached. Dependency
caches are saved immediately after each successful installation, so a later
dependency or numerical failure does not discard those builds. For local work,
`run.py dependencies --dependency NAME` selects a single dependency.
Compiler caches are also saved after ordinary build or test failures, but not
after cancellation. Cache transfers are best-effort; build and test failures
remain failures regardless of cache outcomes.

Manual runs accept `cache_namespace` (default `v1`). Use a new value for a cold
run, then repeat with the same value and explicit dependency SHAs to verify warm
reuse. This isolates acceptance runs without deleting existing caches. A changed
toolchain identity can legitimately cause a cache miss.

Diagnostics live in `.ci/reports/CONFIGURATION/` and are uploaded even on failure,
with seven-day retention. They include dependency revisions, toolchain details,
CMake settings, compile commands, build/test logs, JUnit results, optional-feature
selection, target inventories and exclusions, installed-consumer logs, and
compiler-cache statistics.
Each job's `manifest.json` combines the shared revisions with its toolchain,
architecture, dependency build options, and cache keys.
Tests run sequentially with two OpenMP threads and a five-minute default timeout;
individual test timeouts retain precedence.

Commands stream output while retaining complete local logs. `status.json` records
which phases ran and whether they passed. `prepare` clears previous reports, and
dependency/check invocations invalidate outdated downstream results while keeping
compiled objects and caches. `report` collects only phases attempted in the
current run, so an earlier passing JUnit report cannot mask a failed rerun.

`packages.json` records installed APT package names, versions, and architectures,
plus Python build-tool versions. Non-Debian local runs record that the APT
inventory is unavailable. These records help reproduce toolchain choices, but
do not guarantee that historical APT packages remain downloadable; they are not
part of cache identities.

## Installed-package coverage

`test-installed-package` installs GenDiL into a temporary prefix, moves it, and
builds a fresh consumer using `find_package(GENDIL CONFIG REQUIRED)` and
`GENDIL::GENDIL`. Dependencies stay at their installed locations. The consumer
checks a small vector operation and verifies the relocated include directory and
unchanged consumer install prefix. GenDiL usage requirements come from the
exported target rather than manual include paths or definitions.

```bash
ctest --test-dir .ci/build/minimal -R '^test-installed-package$' --output-on-failure
```

The GPU consumer is compiled with its device language but never executed here.
GPU jobs select only the `configure`, `compile-failure`, and `package` CTest
labels. The `compile-failure` label is reserved for tests registered through the
corresponding helpers; there are currently no such tests. Configuration checks
do cover expected configuration failures. Existing NVCC exclusions remain visible
in the job summary and artifacts.
A passing CUDA/HIP job means the selected device code compiled; actual device
execution remains a separate validation step.
