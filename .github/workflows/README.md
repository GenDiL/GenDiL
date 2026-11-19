# CI workflow overview

[ci.yml](ci.yml) checks CPU numerical results, GPU compilation, library interfaces,
and installed-package usability. All jobs use standard GitHub-hosted CPU runners;
GPU execution remains separate.

Each run has two stages:

1. **`resolve`** selects shared dependency revisions: MFEM's latest development
   commit and stable Hypre/Caliper releases. Manual runs can override these.
2. **`check`** runs each configuration independently: prepare tools, reuse or build
   dependencies, run checks, and save logs. One failure does not cancel the others.

If dependency discovery fails, `minimal` and any scheduled/manual sanitizer job
still run; the workflow retains the discovery failure. Failed helper tests stop
the matrix before builds start.

| Configuration | What it checks and why |
| --- | --- |
| `gcc`, `clang` | CPU tests with OpenMP, MFEM, Hypre, and Caliper; compiler and interface compatibility. |
| `minimal` | CPU tests without optional libraries or OpenMP; standalone usability. |
| `cuda-nvcc`, `cuda-clang` | Both CUDA frontends, which accept different code; compilation and host checks. |
| `hip` | AMD device compilation and host checks. |
| `analysis` | Independent clang-tidy analysis, even if numerical tests fail. |
| `sanitizers` | CPU tests detecting memory errors and undefined behavior. |

Checks run on PRs targeting `main`, pushes to `main`, weekly, and manually.
Sanitizers run only weekly/manually. New PR runs replace obsolete ones.

Test jobs also check `find_package(GENDIL)` using a separate installed-package
consumer. GPU consumers only compile. Release builds compile examples and
benchmarks without running them. Existing GPU exclusions remain visible in reports.

Most extra YAML manages **caches and failure reports**. Separate library caches
avoid rebuilding unchanged dependencies; saving each immediately preserves that
work if a later step fails. Caching saves time but is optional for correctness.
Compiler caches also retain useful work after ordinary build or test failures.
Manual runs can use a fresh `cache_namespace` to test without existing caches.
The repeated cache steps could move into a composite action, at the cost of
another layer to navigate.

Pinned actions/images limit unexpected tool changes. Shared dependency records
and logs help reproduce failures; diagnostics are kept for seven days, including
on failure. Commands stream progress, and package inventories record the tools
installed after container startup. Local reruns discard outdated result reports.

See [configurations.json](../../scripts/ci/configurations.json) for versions and
the [CI runbook](../../scripts/ci/README.md) for reproduction commands and details.
