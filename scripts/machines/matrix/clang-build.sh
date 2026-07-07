#!/bin/bash
# Build GenDiL on Matrix using Clang as CUDA compiler
#
# Environment variables (with defaults):
#   CLANG_MODULE  - Clang module to load (default: clang/19.1.3)
#   CUDA_MODULE   - CUDA module to load (default: cuda/12.2.2)
#   CMAKE_MODULE  - CMake module to load (default: cmake/3.30.5)
#   CUDA_ARCH     - CUDA architecture (default: 90 for H100)
#   BUILD_DIR     - Build directory (default: build_matrix_clang)
#   INSTALL_DIR   - Install directory (default: install_matrix_clang)

set -e

# Set defaults if not provided
CLANG_MODULE="${CLANG_MODULE:-clang/19.1.3}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2.2}"
CMAKE_MODULE="${CMAKE_MODULE:-cmake/3.30.5}"
CUDA_ARCH="${CUDA_ARCH:-90}"
BUILD_DIR="${BUILD_DIR:-build_matrix_clang}"
INSTALL_DIR="${INSTALL_DIR:-install_matrix_clang}"

echo "Building GenDiL with Clang-CUDA"
echo "  Clang:   ${CLANG_MODULE}"
echo "  CUDA:    ${CUDA_MODULE}"
echo "  CMake:   ${CMAKE_MODULE}"
echo "  Arch:    sm_${CUDA_ARCH}"
echo "  Build:   ${BUILD_DIR}"
echo "  Install: ${INSTALL_DIR}"
echo ""

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

module purge && module load "${CLANG_MODULE}" "${CUDA_MODULE}" "${CMAKE_MODULE}"

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX="../${INSTALL_DIR}" \
      -D USE_CUDA=ON \
      -D CMAKE_CXX_COMPILER=clang++ \
      -D CMAKE_CUDA_COMPILER=clang++ \
      -D CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      ..

cmake --build . -j 16
ctest --output-on-failure
cmake --install .

cd ..
