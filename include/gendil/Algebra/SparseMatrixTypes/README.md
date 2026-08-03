# Sparse Matrix Types

GenDiL provides owning sparse matrices, synchronized host/device storage, and
backend-specific sparse matrix-vector products. Include
`gendil/Algebra/SparseMatrixTypes/sparsematrixtypes.hpp` for all sparse types,
or use the format-specific umbrella header when only one representation is
needed.

## Formats and public headers

| Format | Public header | Purpose |
| --- | --- | --- |
| BSR | `BSR/bsrmatrix.hpp` | Block sparse row storage and application |
| SGBSR | `SGBSR/sgbsrmatrix.hpp` | BSR application composed with finite-element gather/scatter maps |
| RawCOO | `COO/rawcoo.hpp` | Unsorted triplet storage used during assembly |
| COO | `COO/coomatrix.hpp` | Canonical coordinate storage and application |
| CSR | `CSR/csrmatrix.hpp` | Compressed sparse row storage and application |
| CSC | `CSC/cscmatrix.hpp` | Compressed sparse column storage and application |
| HypreCSR | `HypreCSR/hyprecsrmatrix.hpp` | Optional Hypre-compatible CSR storage and application |

RawCOO is assembly-only. BSR, SGBSR, COO, CSR, CSC, and HypreCSR support
sparse matrix-vector application.

## Ownership

`BSRMatrix`, `COOMatrix`, `CSRMatrix`, and `CSCMatrix` are move-only owners.
Their host/device allocations are released automatically, and ownership
transfer is explicit through `std::move`. `RawCOOTripletBuffer` and the RawCOO
assembly layout follow the same model. HypreCSR composes an owning CSR matrix,
while SGBSR composes an owning BSR matrix.

Matrix metadata remains public. Depending on the format, this includes fields
such as `num_rows`, `num_cols`, `nnz`, `block_rows`, and `num_blocks`. Each
allocated array is a public `SyncHostDeviceArray` containing its allocation,
logical size, and host/device validity state. These fields are a low-level
escape hatch; normal code should acquire a matrix view before reading or
modifying storage.

## Synchronized views

`BSRMatrixView`, `COOMatrixView`, `CSRMatrixView`, `CSCMatrixView`, and
`RawCOOTripletView` are trivially copyable, non-owning raw-pointer views. They
can be captured by host or device kernels but must not outlive their owner.
Const qualification on the view template arguments expresses whether values
and structural indices are mutable.

Every canonical matrix provides the following free view functions:

| Access | Host | Device | Semantics |
| --- | --- | --- | --- |
| Read | `GetHostReadView` | `GetDeviceReadView` | Synchronize to the selected side and return const pointers |
| Read/write | `GetHostReadWriteView` | `GetDeviceReadWriteView` | Preserve current contents, then make the selected side authoritative |
| Write | `GetHostWriteView` | `GetDeviceWriteView` | Skip synchronization and make the selected side authoritative |

The corresponding `GetKernel*View<OnDevice>` functions select the memory side
at compile time. BSR additionally provides values-only read/write and write
views. These expose mutable values but const row offsets and column indices,
which is the appropriate interface for matrix assembly and coefficient
updates. `Sync(matrix)` makes every initialized matrix array current on both
sides.

For example:

```cpp
auto matrix = MakeCSRMatrix<Real, GlobalIndex>(rows, cols, nnz);
auto host = GetHostWriteView(matrix);
host.row_ptr[0] = 0;
// Populate host.col_ind, host.values, and the remaining row pointers.

const auto device = GetDeviceReadView(matrix); // Synchronizes lazily.
DeviceLoop(matrix.num_rows, [=] GENDIL_HOST_DEVICE (GlobalIndex row) {
   // device.row_ptr, device.col_ind, and device.values are raw pointers.
});
```

Synchronization is tracked independently for every allocation. Direct array
access must first use `ReadHost`, `ReadWriteHost`, `WriteHost`, or the
corresponding device function on that array. Writes through an unprepared raw
pointer are untracked and unsupported; stale host and device copies are never
merged.

Mutable views synchronize storage but do not discover or invalidate vendor
SpMV plans. Modifying matrix values in an unchanged allocation preserves a
cached plan. Before changing dimensions, offsets, indices, block structure, or
allocations, reset the stored vendor backend and every explicit vendor backend
previously used with that matrix. See the [vendor backend documentation](VendorSparse/README.md)
for the complete cache contract.

## Matrix-vector application

Apply-capable matrices use a common free-function interface:

```cpp
Apply(matrix, x, y);              // y = A*x
Apply(backend, matrix, x, y);     // Explicit backend.
ApplyAdd(matrix, x, y);           // y += A*x
ApplyAdd(backend, matrix, x, y);  // Explicit backend.
```

`matrix(x, y)` is an overwrite-only convenience forwarding to `Apply`.
GenDiL and MFEM vectors may be used, including mixed input/output vector
classes when both support the memory space required by the selected backend.

CPU builds default to `Host*Backend`. CUDA and HIP builds default canonical
BSR, COO, CSR, and CSC owners to the corresponding `VendorDevice*Backend`.
`Host*Backend` and `NativeDevice*Backend` remain explicitly selectable.
HypreCSR retains its Hypre host/device default and can forward explicitly to a
CSR backend. Assembly applies an additional capability-aware policy for BSR;
see the [matrix assembly documentation](../../FiniteElementMethod/MatrixAssembly/README.md).

For an empty matrix, `Apply` zeros the output and `ApplyAdd` preserves it. A
vendor backend records a trivial execution and does not retain an initialized
plan for that operation.

## SGBSR

`SGBSRMatrix` publicly composes `bsr_matrix`, `trial_gather`, `test_scatter`,
and reusable `x_bsr` and `y_bsr` workspaces. Free `Apply` and `ApplyAdd`
functions run gather, BSR SpMV, and scatter using the explicitly selected or
stored BSR backend. One SGBSR object must not be applied concurrently because
its workspaces are reused.

Host backends run gather/scatter through host vector access and use OpenMP
when enabled. Native and vendor device backends keep gather, SpMV, and scatter
on the device. Device H1 restrictions must provide a valid device index map.
Shared H1 DoFs are accumulated atomically on host and device, so parallel
results are equivalent within tolerance but need not be bitwise reproducible.

## Further documentation

- [Vendor sparse backends](VendorSparse/README.md)
- [Sparse matrix assembly](../../FiniteElementMethod/MatrixAssembly/README.md)
- [API migration guide](https://github.com/GenDiL/GenDiL/blob/main/scripts/migration/README.md)
