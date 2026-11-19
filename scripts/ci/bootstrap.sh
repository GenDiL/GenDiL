#!/usr/bin/env bash
# Prepare the compiler and build tools for one GenDiL CI configuration.
# Run as root inside the matching Ubuntu container from configurations.json;
# setup requires network access to the package repositories.
#
# Example, from the repository root inside the container:
#   bash scripts/ci/bootstrap.sh minimal
#
# Installs system packages with APT and CMake/Ninja in /opt/gendil-ci-tools.
# GitHub Actions adds that environment to PATH for later steps. Outside Actions,
# run this in the calling shell afterward:
#   export PATH=/opt/gendil-ci-tools/bin:$PATH
#
# run.py drives the subsequent build/check phases; dependencies.py builds
# MFEM, Hypre, and Caliper. See README.md for the complete reproduction sequence.
set -euo pipefail

configuration=${1:?Usage: bootstrap.sh CONFIGURATION}
export DEBIAN_FRONTEND=noninteractive
packages=(git ca-certificates python3 python3-venv ccache make)
# Match the configuration's compiler family and its OpenMP/sanitizer runtimes.
# CUDA and ROCm toolkits come from the container images; GPU cases below add
# host compilers and runtimes needed alongside those toolkits.
case "$configuration" in
  gcc|minimal) packages+=(gcc-14 g++-14) ;;
  clang|analysis)
    packages+=(clang-18 clang-tidy-18 libomp-18-dev) ;;
  sanitizers)
    packages+=(clang-18 clang-tidy-18 libomp-18-dev libclang-rt-18-dev) ;;
  cuda-nvcc) packages+=(gcc-12 g++-12) ;;
  cuda-clang) packages+=(gcc-12 g++-12 clang-19 libomp-19-dev) ;;
  hip) packages+=(gcc-12 g++-12 libomp-dev) ;;
  *) echo "Unknown CI configuration: $configuration" >&2; exit 2 ;;
esac
apt-get update
if [[ "$configuration" == cuda-clang ]]; then
  # The CUDA 12.4.1 image uses Jammy. Match the Clang 19 family used by the
  # Matrix machine build, obtaining it from LLVM's signed Jammy repository.
  apt-get install -y --no-install-recommends ca-certificates curl gnupg
  curl --fail --show-error --location --retry 3 \
    https://apt.llvm.org/llvm-snapshot.gpg.key -o /tmp/gendil-llvm-key.asc
  # Verify the expected LLVM key before trusting it, then scope it to this source.
  fingerprint=$(gpg --show-keys --with-colons /tmp/gendil-llvm-key.asc | awk -F: '$1 == "fpr" { print $10; exit }')
  [[ "$fingerprint" == 6084F3CF814B57C1CF12EFD515CF4D18AF4F7421 ]]
  gpg --batch --yes --dearmor -o /usr/share/keyrings/gendil-llvm.gpg /tmp/gendil-llvm-key.asc
  echo 'deb [signed-by=/usr/share/keyrings/gendil-llvm.gpg] https://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main' \
    > /etc/apt/sources.list.d/gendil-llvm.list
  apt-get update
fi
# APT resolves available package versions; run.py records the installed inventory.
apt-get install -y --no-install-recommends "${packages[@]}"
# Pin CMake/Ninja across images without replacing the system Python packages.
python3 -m venv /opt/gendil-ci-tools
/opt/gendil-ci-tools/bin/pip install --disable-pip-version-check \
  cmake==3.30.5 ninja==1.11.1.1
# GITHUB_PATH affects subsequent workflow steps, not this process's parent shell.
if [[ -n ${GITHUB_PATH:-} ]]; then
  echo /opt/gendil-ci-tools/bin >> "$GITHUB_PATH"
fi
