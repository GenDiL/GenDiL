"""Report which targets and tests a CI configuration includes.

run.py calls request(build) before CMake configuration, then collect(build,
reports, cfg) afterward. CMake's file API supplies target/compiler information;
CTest's show-only mode supplies test registrations and the host-only selection
used by GPU jobs. Neither step builds targets or executes test programs.

The resulting coverage.json and coverage.txt describe configured scope, not
source-line coverage or successful compilation/execution. Build logs and JUnit
reports provide those later results.
"""

import json
from pathlib import Path
import re
import subprocess


# Shared with run.py's GPU CTest command: matching tests must work without a GPU.
SAFE_TEST_LABELS = "^(configure|compile-failure|package)$"
EXPERIMENT = "test-cell-iterator-filtered-sync-experiment"


def request(build):
    """Write file-API queries for the next configure; safe to repeat per attempt."""
    query = Path(build) / ".cmake/api/v1/query/client-gendil-ci"
    query.mkdir(parents=True, exist_ok=True)
    for name in ("codemodel-v2", "toolchains-v1"):
        (query / name).touch()


def read_models(build):
    """Read the latest CMake reply and require the model versions consumed below."""
    reply = Path(build) / ".cmake/api/v1/reply"
    indexes = sorted(reply.glob("index-*.json"))
    if not indexes:
        raise RuntimeError("Missing CMake file-API reply; request coverage before configuring")
    index = json.loads(indexes[-1].read_text())
    models = {}
    for kind, major in (("codemodel", 2), ("toolchains", 1)):
        references = [item for item in index["objects"]
                      if item["kind"] == kind and item["version"]["major"] == major]
        if len(references) != 1:
            raise RuntimeError(f"Expected one CMake {kind} v{major} reply")
        models[kind] = json.loads((reply / references[0]["jsonFile"]).read_text())
    return reply, models


def test_inventory(build, reports, labels=None):
    """Discover tests without running them, optionally filtering by label regex.

    The caller creates reports first. Save CTest's raw stdout/stderr before
    raising on a nonzero discovery result, so a failed inventory is diagnosable.
    """
    command = ["ctest", "--test-dir", str(build), "--show-only=json-v1"]
    if labels:
        command += ["-L", labels]
    result = subprocess.run(command, capture_output=True, text=True)
    stem = "ctest-gpu-selection" if labels else "ctest-inventory"
    (reports / (stem + ".json")).write_text(result.stdout)
    (reports / (stem + ".log")).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"CTest inventory failed ({result.returncode}); see {stem}.log")
    data = json.loads(result.stdout)
    return [{"name": test["name"], "labels": next(
        (prop["value"] for prop in test.get("properties", []) if prop["name"] == "LABELS"), [])}
        for test in data["tests"]]


def collect(build, reports, cfg):
    """Inspect a configured build using an entry from configurations.json.

    Write both inventory formats, then raise if required target groups are empty
    or GPU compiler/language/host-check selection is inconsistent. Configured
    EXCLUDE_FROM_ALL targets can appear here even though a normal build skips them.
    """
    build, reports = Path(build), Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    reply, models = read_models(build)
    compilers = {item["language"]: item.get("compiler", {})
                 for item in models["toolchains"]["toolchains"]}
    groups = {name: [] for name in ("tests", "examples", "benchmarks")}
    source_root = Path(models["codemodel"]["paths"]["source"])
    for configuration in models["codemodel"]["configurations"]:
        for reference in configuration["targets"]:
            target = json.loads((reply / reference["jsonFile"]).read_text())
            if target["type"] != "EXECUTABLE":
                continue
            source = Path(target["paths"]["source"])
            if source.is_absolute():
                try:
                    source = source.relative_to(source_root)
                except ValueError:
                    continue
            category = source.parts[0] if source.parts else ""
            # Classify GenDiL executables by their source directory; ignore TPL targets.
            if category not in groups:
                continue
            compile_groups = target.get("compileGroups", [])
            compiled_sources = [{"path": item["path"],
                                 "language": compile_groups[item["compileGroupIndex"]]["language"]}
                                for item in target.get("sources", []) if "compileGroupIndex" in item]
            groups[category].append({"name": target["name"], "configuration": configuration["name"],
                                     "languages": sorted({item["language"] for item in compiled_sources}),
                                     "sources": compiled_sources})
    for targets in groups.values():
        targets.sort(key=lambda item: (item["name"], item["configuration"]))
    registered = test_inventory(build, reports)
    device_language = cfg["backend"].upper() if cfg["backend"] != "cpu" else None
    selected = test_inventory(build, reports, SAFE_TEST_LABELS) if device_language else None
    configure_log = reports / "configure.log"
    exclusions = [line.strip() for line in configure_log.read_text().splitlines()
                  if re.search(r"Skipping (?:test|benchmark) ", line)] if configure_log.exists() else []
    errors = []
    for category in ("tests", "examples") + (("benchmarks",) if cfg.get("benchmarks") else ()):
        if not groups[category]:
            errors.append(f"No {category} executable targets configured")
    if not registered:
        errors.append("No CTests registered")
    if device_language:
        expected_id = cfg.get("device_compiler_id")
        actual_id = compilers.get(device_language, {}).get("id")
        if not actual_id or (expected_id and actual_id != expected_id):
            errors.append(f"Expected {device_language} compiler {expected_id or 'to be enabled'}, found {actual_id}")
        for targets in groups.values():
            for target in targets:
                if target["languages"] != [device_language]:
                    errors.append(f"GPU target {target['name']} uses {target['languages']}, expected {device_language}")
        if not selected:
            errors.append("No host-only GPU checks selected")
        if any(not any(re.fullmatch(SAFE_TEST_LABELS, label) for label in test["labels"])
               for test in selected):
            errors.append("GPU CTest selection contains a test without a permitted host-only label")
    # Inventory cannot prove default-build membership or test success.
    notes = ["Configured executable inventory only; this report does not establish compilation or execution success.",
             "CMake's file API does not expose default-build membership (EXCLUDE_FROM_ALL)."]
    if any(target["name"] == EXPERIMENT for target in groups["tests"]):
        notes.append(f"{EXPERIMENT} is intentionally excluded from the normal build.")
    inventory = {"schema_version": 1, "backend": cfg["backend"], "compilers": compilers,
                 "targets": groups, "ctest_registered": registered,
                 "ctest_gpu_host_only_selection": selected, "exclusions": exclusions,
                 "notes": notes, "errors": errors}
    (reports / "coverage.json").write_text(json.dumps(inventory, indent=2) + "\n")
    lines = ["Configured compilation scope (not build/run results)"]
    lines += [f"{language}: {compiler.get('id', 'unknown')} {compiler.get('version', '')} "
              f"({compiler.get('path', 'unknown path')})" for language, compiler in sorted(compilers.items())]
    lines += [f"{name}: {len(targets)} executable targets" for name, targets in groups.items()]
    lines.append(f"CTest registrations: {len(registered)}")
    if selected is not None:
        lines.append(f"GPU host-only CTest selection: {len(selected)} ({SAFE_TEST_LABELS})")
    lines += notes + ["Exclusions:"] + (exclusions or ["None reported by configuration."])
    if errors:
        lines += ["Coverage errors:"] + errors
    (reports / "coverage.txt").write_text("\n".join(lines) + "\n")
    if errors:
        raise RuntimeError("Invalid CI coverage; see coverage.json: " + "; ".join(errors))
    return inventory
