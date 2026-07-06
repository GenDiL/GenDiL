[![DOI](https://zenodo.org/badge/DOI/10.11578/dc.20250507.2.svg)](https://doi.org/10.11578/dc.20250507.2)
[![CI](https://github.com/GenDiL/GenDiL/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/GenDiL/GenDiL/actions)

# GenDiL

<!-- toc -->
- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Benchmarks, tests, and examples](#benchmarks-tests-and-examples)
- [Weak Form DSL](#weak-form-dsl)
- [Citing GenDiL](#citing-gendil)
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
- h-adaptivity and p-adaptivity support
- Matrix-free operator evaluation
- Sparse matrix assembly for BSR, COO, CSC, CSR, and optional HypreCSR formats
- Weak-form expression templates for high-level operator specification

### Third-Party Integrations & Parallelization Support
- Interfaces with the [MFEM](https://github.com/mfem/mfem), [HYPRE](https://github.com/hypre-space/hypre), [RAJA](https://github.com/LLNL/RAJA), and [Caliper](https://github.com/LLNL/Caliper) libraries.
- Support for OpenMP, CUDA, HIP parallelization models

## Getting Started
### Requirements
- C++20 or higher

### Installation
GenDiL is header-only, so the simplest integration path is to add the
`include` folder to your project's include path. For CMake projects, the
recommended workflow is to configure, build, test, and install GenDiL out of
source.

Minimal CMake install:
```sh
cmake -S . -B build \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX="$PWD/install"

cmake --build build --parallel
ctest --test-dir build
cmake --install build
```

OpenMP support is enabled by default. If OpenMP is not available or not
desired, add `-D USE_OPENMP=OFF` when configuring.

With optional integrations:
```sh
cmake -S . -B build-deps \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX="$PWD/install-deps" \
      -D USE_MFEM=ON \
      -D MFEM_DIR=/path/to/mfem/install \
      -D USE_HYPRE=ON \
      -D HYPRE_DIR=/path/to/hypre/install \
      -D USE_RAJA=ON \
      -D RAJA_DIR=/path/to/raja/install \
      -D USE_CALIPER=ON \
      -D caliper_DIR=/path/to/caliper/install

cmake --build build-deps --parallel
ctest --test-dir build-deps
cmake --install build-deps
```

CUDA and HIP support are enabled separately with `-D USE_CUDA=ON` or
`-D USE_HIP=ON`. HypreCSR device support requires a matching CUDA- or
HIP-enabled HYPRE build.

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

## Weak Form DSL
GenDiL weak forms are expression templates built from trial/test fields,
integration domains, and algebraic operators. A typical weak form declares
`TrialSpace` and `TestSpace` symbols, selects `Cells`, `InteriorFacets`, and
`BoundaryFacets` domains, then combines `integrate` with operators such as
`grad`, `jump`, `average`, `upwind`, `dot`, and `Normal`.

For an upwind DG advection operator, the weak form can be written directly from
the cell and interior-facet terms:

```cpp
TrialSpace<"u"> u;
TestSpace<"u"> v;
Cells<"mesh"> cells;
InteriorFacets<"mesh"> interior_facets;

auto beta = MakeVectorCoefficient<"beta", PhysicalCoordinate>(beta_fn);

auto upwind_dg_form =
   integrate(cells, -u * dot(beta, grad(v)))
   + integrate(interior_facets, upwind(beta, u) * jump(v));
```

For SIPDG diffusion, the same DSL captures cell, interior-facet, and
boundary-facet contributions:

```cpp
TrialSpace<"u"> u;
TestSpace<"u"> v;
Cells<"mesh"> cells;
InteriorFacets<"mesh"> interior_facets;
BoundaryFacets<"mesh"> boundary_facets;

auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(mu_fn);
auto tau = MakeCoefficient<"penalty", InverseFacetSize>(tau_fn);

auto sipdg_form =
   integrate(cells, mu * dot(grad(u), grad(v)))
   + integrate(
        interior_facets,
        -average(mu * dot(grad(u), Normal{})) * jump(v)
        + jump(u) * average(mu * dot(grad(v), Normal{}))
        + tau * mu * jump(u) * jump(v))
   + integrate(
        boundary_facets,
        -mu * dot(grad(u), Normal{}) * v
        + u * mu * dot(grad(v), Normal{})
        + tau * mu * u * v);
```

The same weak form can create a matrix-free operator or an assembled sparse
matrix. Here `weak_form` is either form above, and `fe_space` and
`integration_rule` are the finite element space and quadrature rule for the
domain named `"mesh"`:

```cpp
using KernelPolicy = SerialKernelConfiguration;

auto context = MakeWeakFormContext(
   MakeTrialField<"u">(fe_space),
   MakeIntegrationDomain<"mesh">(fe_space));

auto op =
   MakeGenericOperator<KernelPolicy>(
      weak_form,
      context,
      integration_rule);

auto matrix =
   GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
      weak_form,
      context,
      integration_rule);
```

Use `MatrixAssemblyType::COO`, `MatrixAssemblyType::CSR`,
`MatrixAssemblyType::CSC`, or, when GenDiL is configured with HYPRE,
`MatrixAssemblyType::HypreCSR` to assemble other sparse formats.

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
