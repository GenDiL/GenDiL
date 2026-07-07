# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.5] - 2026-07-06

### Added
- Canonical sparse matrix assembly and apply support for RawCOO, COO, CSR, CSC, SGBSR, BSR, and optional HypreCSR formats, including `MatrixAssemblyType` dispatch and default backend selection.
- Initial support for HYPRE.
- Host, OpenMP, and native-device sparse matrix-vector apply paths, sparse apply arithmetic policies, and MFEM-compatible apply overloads.
- Vector H1/CG and vector finite element assembly support through `VectorH1Restriction` and SGBSR gather/scatter.
- Batched kernel configuration and batched operator coverage for L2 projection/error, mass/mass-inverse, linear form, grad-grad, mixed mass, speed-of-light, and global-face advection/speed-of-light paths.
- Weak-form DSL support for `transpose`, `minus`/`plus` trace expressions, and coefficient inputs from named `FieldValue`/`FieldGradient` fields.
- `Partition`, `CellPart`, `InteriorFacePart`, `BoundaryFacePart`, and `MakePartition` data structures for partition-owned cell and face meshes used by mixed finite element spaces and global-face domains.
- `GenericOperator` support for global face meshes, mixed finite element spaces, partitioned domains, tensor-product restrictions, h-adaptivity, unstructured global face connectivity, and MFEM global-face builders including nonconforming point-matrix decoding.
- Dry-run migration scripts under `scripts/migration` for mechanical source updates from `v0.0.3` to `v0.0.4` and from `v0.0.4` to the planned `v0.0.5` API.
- Expanded tests and benchmarks for sparse formats, matrix assembly defaults, batched/global-face paths, mixed spaces, adaptivity, coefficient field inputs, MFEM, Hypre, CUDA/HIP, and face-DoF policies.

### Changed
- Breaking: refactored sparse matrix and matrix assembly headers into format-specific public storage/apply/finalization layers.
- Breaking: refactored `GenericOperator` into cell, local-facet, global-facet, and context components with backend-neutral weak-form traversal.
- Breaking: moved restriction/DoF-layout APIs into the `FiniteElementMethod/Restrictions` structure and moved finite-element orders under shape functions.
- Breaking: renamed weak-form context domain entries from `MakeDomain<Name>(mesh)` to `MakeIntegrationDomain<Name>(space)`, where `space` is either a homogeneous `FiniteElementSpace` or a `Partition`-backed `MixedFiniteElementSpace` for partitioned/global-face domains.
- Breaking: renamed element DoF view helpers from `MakeEVectorView`, `MakeScalarEVectorView`, `MakeVectorEVectorView`, `MakeReadOnlyEVectorView`, `MakeWriteOnlyEVectorView`, and `MakeReadWriteEVectorView` to the corresponding `Make*ElementTensorView` names.
- Breaking: renamed `make_cartesian_interior_face_connectivity` to `MakeCartesianInteriorFaceConnectivity`.
- Breaking: renamed the MFEM local connectivity helper/header from `MakeMeshConnectivity` and `meshconnectivity.hpp` to `MakeMeshLocalConnectivity` and `meshlocalconnectivity.hpp`.
- Breaking: changed the three-argument `GenericAssembly<KernelPolicy>(...)` return form to require an explicit `MatrixAssemblyType` first, such as `GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(...)`; the four-argument in-place form remains available.
- Refactored kernel configuration, thread layout, shared-memory sizing, and benchmark/test CMake configuration for batching and device compatibility.
- Changed `ProductCell` to a struct, standardized selected names to CamelCase, and renamed matrix machine scripts to `nvcc-build.sh`/`clang-build.sh`.
- Exposed mass-inverse tolerance and maximum-iteration controls.

### Removed
- Breaking: removed the old `CellIterator` path.
- Breaking: removed persistent face finite element spaces in favor of partition-owned face meshes for mixed finite element spaces.
- Breaking: removed or relocated legacy top-level sparse matrix, matrix assembly, restriction, DoF-layout, and `GenericOperator` context headers; users should include the umbrella headers or new format/context-specific paths.
- Removed the BSR backend accessor.

### Fixed
- Fixed legacy diffusion operator behavior and avoided unnecessary plus-side Jacobian computation.
- Fixed BSR device apply, MFEM vector pointer access, `ReadDofs`, missing host/device annotations, shared-memory synchronization/arena behavior, and GPU 5D p-adaptivity issues.
- Improved diagnostics and guards for unsupported batched kernels, shared-memory arena errors, threaded aggregate dimensions, thread-layout undercoverage, and invalid layouts.

---

## [0.0.4] - 2026-05-17

### Added
- Added initial sparse matrix assembly support using BSR format.
- Added a high-level operator factory for constructing matrix-free and assembled operators from expression-template weak form specifications.

---

## [0.0.3] - 2026-03-25

### Added
- Support for h-adaptivity.
- Support for p-adaptivity.
- Support for face meshes.
- Mesh concept.
- GlobalFaceMeshConnectivity concept.
- DofToQuadMapping concept.

### Changed
- Switched the project to C++20.
- Renamed `GetFaceNeighborInfo` to `GetLocalFaceInfo`.

### Fixed
- Caliper integration fixes and assorted correctness/cleanup changes included in PR #16.

---

## [0.0.2] - 2025-04-18

### Added
- Support for CUDA and HIP GPUs.

---

## [0.0.1] - 2025-03-18

### Added
- Initial features for discontinuous Galerkin finite element methods.
- Basic testing framework setup and example tests.
- Basic benchmarks framework.
- Instructions for installation and basic usage in the README.
- Basic support for CUDA GPUs.


<!-- TEMPLATE: Future Versions -->

## [X.X.X] - YYYY-MM-DD

### Added
- Description of new features added in this version.

### Changed
- Description of modifications or improvements to existing features.

### Deprecated
- List any features marked for future removal.

### Removed
- List features or components removed from the project.

### Fixed
- Description of bug fixes included in this release.

### Security
- Details of any security improvements.
