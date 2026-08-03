# Vendor Sparse Backends

GenDiL maps canonical sparse storage directly into cuSPARSE and rocSPARSE
descriptors. Vendor application never copies the sparse representation and
never silently dispatches a native GenDiL kernel.

See the parent [sparse matrix documentation](../README.md) for ownership,
views, synchronization, and the common `Apply` interface.

## Backends and defaults

CUDA builds provide `CuSparse{BSR,COO,CSR,CSC}Backend`; HIP builds provide
`RocSparse{BSR,COO,CSR,CSC}Backend`. The active platform also defines the
portable aliases `VendorDevice{BSR,COO,CSR,CSC}Backend`.

Standalone BSR, COO, CSR, and CSC matrices use the active vendor backend by
default in a device build. CPU builds use the corresponding host backend.
Host and native-device implementations remain available explicitly through
`Host*Backend` and `NativeDevice*Backend`. Capability-aware BSR assembly may
select a native backend instead; see the [assembly documentation](../../../FiniteElementMethod/MatrixAssembly/README.md).

## Initialized SpMV state

The first nonempty vendor operation lazily initializes backend-owned state:

- a vendor handle on the default stream;
- sparse and dense descriptors;
- the selected SpMV algorithm and preprocessing state;
- a reusable workspace;
- the currently bound input and output vector pointers.

Repeated compatible operations reuse this state. Changing input or output
allocations only rebinds the dense descriptors. Alternating `Apply` and
`ApplyAdd`, or changing values in place without replacing their allocation,
does not rebuild the sparse plan. Dimensions, types, sparse storage pointers,
layout, format, and algorithm form the plan identity; a change detected during
initialization replaces stale state.

Backend copies preserve their configuration but start with an empty cache.
Backend moves transfer initialized state. `HasCachedPlan()` reports whether a
complete plan is initialized, and `LastExecutionPath()` reports uninitialized,
vendor, or trivial execution. A backend object must not be applied concurrently
from multiple host threads.

Initialization uses stable `alpha=1` and `beta=0` values for sizing and
preprocessing. Execution supplies `alpha=1, beta=0` for `Apply` and
`alpha=1, beta=1` for `ApplyAdd`, so the two public operations share one plan.

## Structural mutation

Value changes do not require cache invalidation when the values allocation is
unchanged. Structural mutation is caller-managed because mutable views cannot
identify every backend that may have cached descriptors for a matrix.

Before changing dimensions, offsets, indices, block structure, or sparse
allocations, call `ResetState()` on the stored backend and every explicit
vendor backend previously applied to the matrix:

```cpp
matrix.backend.ResetState();
external_backend.ResetState();
auto structure = GetHostReadWriteView(matrix);
// Modify structure and keep its metadata consistent with the owner.
```

Reset before mutation because initialized vendor descriptors and outstanding
operations may still refer to the existing storage. View acquisition itself
does not reset vendor state.

## Format and index support

COO, CSR, and CSC storage maps directly to generic vendor descriptors. BSR
preserves row-major or column-major block layout but requires square blocks
for explicit vendor application. Rectangular BSR application reports a
runtime shape error and directs callers to `NativeDeviceBSRBackend`.

Vendor indices must be bit-compatible 32-bit or 64-bit integral types. GenDiL
checks that dimensions and nonzero counts fit the corresponding signed vendor
range before descriptor creation. Representation and vendor API failures are
reported with operation context rather than falling back to native kernels.

## BSR toolkit capabilities

Generic cuSPARSE BSR SpMV is available beginning with CUDA 13.0 Update 1.
GenDiL detects the required `cusparseCreateBsr` API at configuration time.
Explicit cuSPARSE BSR application on an older toolkit produces a dependent
compile-time diagnostic.

rocSPARSE BSR is enabled when `rocsparse_create_bsr_descr` and the generic BSR
SpMV algorithm are available. ROCm 6.3.1 does not provide this API; ROCm 6.4
and newer releases do. An unavailable rocSPARSE BSR path likewise produces a
dependent compile-time diagnostic.

These capability diagnostics apply when the corresponding BSR `Apply` or
`ApplyAdd` template is instantiated, regardless of the matrix's runtime
contents. Typed assembly avoids selecting an unavailable vendor BSR backend.

## Arithmetic and empty matrices

Uniform `float` and `double` matrix/vector arithmetic is supported. A `float`
matrix with `double` input, output, and computation is supported by cuSPARSE
beginning with CUDA 12.5 Update 1 (toolkit build 12.5.82). Older CUDA releases
and rocSPARSE reject this combination at compile time. Other unsupported
value, vector, computation, or index types also receive focused compile-time
diagnostics.

For a supported backend/type combination, empty matrices use a shared trivial
path without creating descriptors or a plan. `Apply` zeros the output;
`ApplyAdd` leaves it unchanged. Trivial execution resets any previous cached
state and records `VendorSparseExecutionPath::Trivial`.
