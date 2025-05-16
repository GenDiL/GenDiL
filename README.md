[![DOI](https://zenodo.org/badge/DOI/10.11578/dc.20250507.2.svg)](https://doi.org/10.11578/dc.20250507.2)
[![CI](https://github.com/GenDiL/GenDiL/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/GenDiL/GenDiL/actions)

# GenDiL

<!-- toc -->
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
<!-- tocstop -->

GenDiL stands for Generic Discretization Library.

## Overview
GenDiL is a header-only C++ library providing flexible and efficient discretization tools for partial differential equations (PDEs) through a generic programming approach. The library aims to bring advanced mathematical concepts into practical use, leveraging modern C++ for high-performance scientific computing.

<!-- One of the key aspects of GenDiL is its emphasis on generic programming. Algorithms are expressed using functions, allowing for flexibility and reusability. -->
<!-- Mathematical concepts are represented through their interfaces, rather than relying on rigid data structures. This approach ensures that the library remains close to the mathematical formulation, making it easier to formalize and reason about the code. By focusing on functions and interfaces, GenDiL enables a clean and modular design that is intrinsically closer to mathematical specifications, providing an intuitive development experience for scientific computing. Additionally, this approach facilitates seamless interfacing with other libraries, as the abstraction through interfaces allows for better compatibility and integration without being tied to specific data structures. -->

## Features
### Meshing
- Support for structured and unstructured mesh representations
- High-dimension mesh construction through Cartesian product

### Finite Element Method
- Continuous and Discontinuous Galerkin methods
- Arbitrary dimension hypercube finite elements
- Anisotropic polynomial orders and quadrature rules
- Matrix-free operator evaluation

### Third-Party Integrations & Parallelization Support
- Interfaces with the [MFEM](https://github.com/mfem/mfem), [RAJA](https://github.com/LLNL/RAJA), and [Caliper](https://github.com/LLNL/Caliper) libraries.
- Support for OpenMP, CUDA, HIP parallelization models

## Getting Started
### Requirements
- C++17 or higher

### Installation
Simply add the `include` folder to your project's include path. The GenDiL project also ships with `CMakeLists.txt` files for building with CMake. The `scripts` directory contains examples scripts for building and using GenDiL.

The following script will use CMake to configure, build, run the tests, and install the library:
```sh
mkdir -p build
cd build

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install \
      -D USE_MFEM=ON \
      -D MFEM_DIR=../mfem/build \
      ..

make -j 8 && make test && make install

cd ..
```

### Usage
Include the main header file in your project:
```cpp
#include <gendil/gendil.hpp>
```

When using CMake for your project, use `find_package` to import GenDiL:
```
find_package(GENDIL REQUIRED HINTS "${GENDIL_DIR}" "${GENDIL_DIR}/share/gendil/cmake" NO_DEFAULT_PATH)
```

## Benchmarks, tests, and examples
Executables are organized in three categories: benchmarks, tests, and examples.
- Examples show standard use cases of the library.
- Tests verify the correctness of the library.
- Benchmarks measure the performance of the library.

## Citing GenDiL

If you use GenDiL in your publications or presentations, please cite it as follows:

```bibtex
@misc{doecode_154944,
  title        = {GenDiL},
  author       = {Dudouit, Yohann and Holec, Milan and Rotem, Amit Y.},
  abstractNote = {The GenDiL library is a collection of C++ software abstractions designed to discretize and solve partial differential equations (PDEs) for high‐performance computing (HPC) applications. Its primary focus is on modern C++ generic programming, which helps ensure portability across various hardware architectures.},
  doi          = {10.11578/dc.20250507.2},
  url          = {https://doi.org/10.11578/dc.20250507.2},
  howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20250507.2}},
  year         = {2025},
  month        = {march}
}
```

## Contributing

All new contributions must be made under the BSD-3-Clause license.

## License
GenDiL is distributed under the terms of the BSD-3-Clause license.

See [LICENSE](./LICENSE), [COPYRIGHT](./COPYRIGHT), and [NOTICE](./NOTICE) for details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-2003681
