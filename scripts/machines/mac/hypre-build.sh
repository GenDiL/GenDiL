#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"

HYPRE_SOURCE_DIR="${HYPRE_SOURCE_DIR:-${GENDIL_ROOT}/../hypre}"
HYPRE_BUILD_DIR="${HYPRE_BUILD_DIR:-${HYPRE_SOURCE_DIR}/build-gendil}"
HYPRE_INSTALL_DIR="${HYPRE_INSTALL_DIR:-${HYPRE_SOURCE_DIR}/install-gendil}"

GENDIL_BUILD_DIR="${GENDIL_BUILD_DIR:-${GENDIL_ROOT}/build-hypre}"
GENDIL_INSTALL_DIR="${GENDIL_INSTALL_DIR:-${GENDIL_ROOT}/install-hypre}"

JOBS="${JOBS:-8}"

if [[ -z "${OpenMP_ROOT:-}" ]]; then
   if ! command -v brew >/dev/null 2>&1; then
      echo "error: Homebrew is required to locate libomp. Install Homebrew/libomp or set OpenMP_ROOT." >&2
      exit 1
   fi
   export OpenMP_ROOT="$(brew --prefix)/opt/libomp"
else
   export OpenMP_ROOT
fi

if [[ ! -d "${OpenMP_ROOT}" ]]; then
   echo "error: OpenMP_ROOT does not exist: ${OpenMP_ROOT}" >&2
   echo "       Install libomp with Homebrew or set OpenMP_ROOT explicitly." >&2
   exit 1
fi

HYPRE_CONFIG="${HYPRE_INSTALL_DIR}/lib/cmake/HYPRE/HYPREConfig.cmake"

if [[ ! -f "${HYPRE_CONFIG}" ]]; then
   if [[ ! -f "${HYPRE_SOURCE_DIR}/src/CMakeLists.txt" ]]; then
      echo "error: Hypre install was not found and Hypre source is missing:" >&2
      echo "       install: ${HYPRE_CONFIG}" >&2
      echo "       source:  ${HYPRE_SOURCE_DIR}/src/CMakeLists.txt" >&2
      exit 1
   fi

   echo "Hypre install not found. Building Hypre in ${HYPRE_BUILD_DIR}"
   rm -rf "${HYPRE_BUILD_DIR}"
   cmake -S "${HYPRE_SOURCE_DIR}/src" -B "${HYPRE_BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${HYPRE_INSTALL_DIR}" \
      -DHYPRE_ENABLE_MPI=OFF \
      -DHYPRE_ENABLE_OPENMP=ON \
      -DHYPRE_ENABLE_BIGINT=OFF \
      -DHYPRE_ENABLE_MIXEDINT=OFF \
      -DHYPRE_ENABLE_SINGLE=OFF \
      -DHYPRE_ENABLE_LONG_DOUBLE=OFF \
      -DHYPRE_BUILD_TESTS=OFF \
      -DOpenMP_ROOT="${OpenMP_ROOT}"

   cmake --build "${HYPRE_BUILD_DIR}" -j "${JOBS}"
   cmake --install "${HYPRE_BUILD_DIR}"
else
   echo "Using existing Hypre install: ${HYPRE_INSTALL_DIR}"
fi

cmake -S "${GENDIL_ROOT}" -B "${GENDIL_BUILD_DIR}" \
   -DCMAKE_BUILD_TYPE=Release \
   -DUSE_OPENMP=ON \
   -DUSE_HYPRE=ON \
   -DHYPRE_ROOT="${HYPRE_INSTALL_DIR}" \
   -DOpenMP_ROOT="${OpenMP_ROOT}" \
   -DCMAKE_INSTALL_PREFIX="${GENDIL_INSTALL_DIR}"

cmake --build "${GENDIL_BUILD_DIR}" -j "${JOBS}"
ctest --test-dir "${GENDIL_BUILD_DIR}" --output-on-failure
cmake --install "${GENDIL_BUILD_DIR}"
