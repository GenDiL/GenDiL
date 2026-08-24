# Sparse Matrix Assembly

GenDiL assembles weak forms into BSR, SGBSR, RawCOO, COO, CSR, CSC, and
optional HypreCSR storage. The primary entry point is:

```cpp
auto matrix =
   GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
      weak_form,
      weak_form_context,
      integration_rule);
```

An overload accepting an explicit backend is available for formats that store
an apply backend. `GenericElementBlockDiagonalAssembly` provides the analogous
element-block-diagonal path. RawCOO is the intermediate unsorted triplet
representation; canonical COO, CSR, and CSC finalization sorts and reduces it.

See the [sparse matrix documentation](../../Algebra/SparseMatrixTypes/README.md)
for ownership, views, synchronization, and matrix application.

## Default backends

CPU assembly defaults BSR, SGBSR, COO, CSR, and CSC matrices to host backends.
CUDA and HIP assembly defaults COO, CSR, and CSC to the active vendor backend.
HypreCSR uses the Hypre device backend only when both GenDiL and Hypre have
matching device support; otherwise it uses the Hypre host backend. RawCOO is
assembly-only and has no apply backend.

BSR and SGBSR make a more specific choice. `MakeDefaultBSRBackend` inspects the
trial and test spaces named by the weak form and obtains their local DoF
counts from the weak-form context:

- a CPU build selects `HostBSRBackend`;
- a device build selects `VendorDeviceBSRBackend` only for square blocks when
  generic vendor BSR SpMV is available;
- rectangular blocks, or a toolkit without generic BSR SpMV, select
  `NativeDeviceBSRBackend`.

The selected type is fixed when the assembled matrix is returned. Explicit
backend arguments override this default policy; explicit vendor BSR
application remains subject to the capability and square-block requirements
described in the [vendor documentation](../../Algebra/SparseMatrixTypes/VendorSparse/README.md).

## Assembly memory placement

The `KernelPolicy` determines whether assembly runs in host or device memory.
Format-specific entry points acquire an appropriate non-owning assembly target
before calling generic dispatch. BSR and SGBSR use a values-only
`GetKernelValuesReadWriteView` so row offsets and column indices remain const
while coefficients are accumulated. This preserves reusable vendor sparsity
preprocessing.

RawCOO assembly constructs a `RawCOOAssemblyTarget` from synchronized triplet
and layout storage. Device assembly synchronizes before locally owned layout
storage is destroyed. Sorting and canonical finalization explicitly acquire
the host or device views they require rather than guessing a synchronization
direction.

Assembly leaves the side selected by the kernel policy authoritative. A later
host or device consumer synchronizes lazily through the matrix view interface.

## Weak-form contexts and domains

A weak-form context binds named trial, test, and coefficient fields to the
named domains used by a weak form. Pass a mesh for a homogeneous domain, or a
partition for a partitioned global-face domain:

```cpp
auto global_context = MakeWeakFormContext(
   MakeTrialField<"u">(trial_mixed),
   MakeTestField<"v">(test_mixed),
   MakeFiniteElementField<"conductivity">(
      conductivity_mixed,
      conductivity_mixed_dofs),
   MakeIntegrationDomain<"skeleton">(partition));
```

Matrix-free operators support mesh and partition domain kinds. Full RawCOO
assembly supports partition cell, boundary-face, same-part interior-face, and
cross-part interior-face terms. COO, CSR, CSC, and process-local HypreCSR obtain
the same support through RawCOO finalization. Homogeneous assembly contexts
bind `MakeIntegrationDomain<Name>(mesh)`.

SGBSR cell assembly accepts independently supported trial gather and test
scatter mappings, including rectangular pairs. The same independently valid
mappings are accepted for boundary and interior facet terms; potentially
sharing restrictions use zeroing plus atomic accumulation during scatter. The
default `RestrictionGatherToBsr<Space>` and
`RestrictionScatterFromBsr<Space>` mappings require the restriction
to provide `ValidateRestrictionMemoryAccess<OnDevice>(restriction,
num_elements)` through normal lookup or ADL. This operation validates any
borrowed mapping storage for the memory space selected by the matvec backend;
contiguous restrictions implement it as a no-op, while vector and tensor
products recurse into their children. Current mappings require one unit-weight
entry per row. Multi-entry mappings can retain the element-block matrix by
gathering through the restriction and scattering through its adjoint once a
general weight protocol is available.

RawCOO and its derived COO/CSR/CSC formats support conforming mesh-local and
partition global-facet terms for statically one-entry, unit-weight,
reference-addressable completed restrictions. Partition assembly emits one
monolithic matrix in the trial/test algebraic extents and preserves explicit
placement, automatic L2 direct sums, and shared H1 aliases as duplicate raw
coordinates. Each interior-face part owns exact `--`, `-+`, `+-`, and `++`
segments; canonical finalization reduces duplicates within and across parts.

Nonconforming global facets support scalar and componentwise
`VectorShapeFunctions` value/gradient transformations. Geometric Piola,
H(div), H(curl), and de Rham-style transformations remain unsupported. The
partition output is process-local; HypreCSR continues to use
`hypre_MPI_COMM_SELF`. RawCOO execution may use batched device policies through
the existing `BlockLoop`; BSR/SGBSR batching and partition assembly remain
outside this path.
