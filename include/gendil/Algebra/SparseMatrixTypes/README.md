# Sparse matrix types

Store assembled linear operators and evaluate matrix-vector products without storing every zero coefficient.
GenDiL matrices manage their storage and can apply the same operator on the host or a configured device backend.

## A small CSR example

This matrix has two rows, three columns, and four stored entries. Applying it
to `x = (1,2,3)` gives `y = (-1,18)`:

```cpp
#include <gendil/Algebra/SparseMatrixTypes/CSR/csrmatrix.hpp>
#include <gendil/Algebra/Vector/access.hpp>
using namespace gendil;

// A = [ 2  0 -1 ]
//     [ 0  3  4 ]
constexpr GlobalIndex num_rows = 2;
constexpr GlobalIndex num_cols = 3;
constexpr GlobalIndex num_entries = 4;
auto matrix = MakeCSRMatrix<Real, GlobalIndex>(num_rows, num_cols, num_entries);
auto host = GetHostWriteView(matrix);
host.row_ptr[0] = 0; host.row_ptr[1] = 2; host.row_ptr[2] = 4;
host.col_ind[0] = 0; host.col_ind[1] = 2;
host.col_ind[2] = 1; host.col_ind[3] = 2;
host.values[0] = 2; host.values[1] = -1;
host.values[2] = 3; host.values[3] = 4;

Vector x(num_cols), y(num_rows);
auto input = WriteHostVector(x);
input[0] = 1; input[1] = 2; input[2] = 3;

matrix(x, y); // Replaces y with A*x; no initialization of y is needed.
const auto result = ReadHostVector(y);
// result[0] == -1, result[1] == 18.
```

CSR means *compressed sparse row*. `row_ptr[r]` and `row_ptr[r+1]` delimit
row `r` in the column-index and value arrays. Indices are zero-based; each row's
columns are sorted and unique. `MakeCSRMatrix` allocates storage; the example
initializes every array before application. The write view provides access
without copying matrix entries.

In finite-element applications, normally obtain the matrix from
[GenericAssembly](../../FiniteElementMethod/MatrixAssembly/README.md) rather
than fill these arrays yourself.

## Choosing a representation

| Format | When to use it | Public header |
| --- | --- | --- |
| CSR | A globally assembled matrix organized by rows | [CSR/csrmatrix.hpp](CSR/csrmatrix.hpp) |
| CSC | A globally assembled matrix organized by columns | [CSC/cscmatrix.hpp](CSC/cscmatrix.hpp) |
| COO | Explicit row/column/value entries, sorted and duplicate-reduced | [COO/coomatrix.hpp](COO/coomatrix.hpp) |
| BSR | A sparse pattern of dense blocks, such as element-local matrices | [BSR/bsrmatrix.hpp](BSR/bsrmatrix.hpp) |
| SGBSR | Apply BSR blocks through finite-element gather/scatter maps | [SGBSR/sgbsrmatrix.hpp](SGBSR/sgbsrmatrix.hpp) |
| RawCOO | Collect unsorted, possibly duplicate assembly contributions | [COO/rawcoo.hpp](COO/rawcoo.hpp) |
| HypreCSR | Optional Hypre application using GenDiL-owned CSR storage | [HypreCSR/hyprecsrmatrix.hpp](HypreCSR/hyprecsrmatrix.hpp) |

RawCOO is assembly storage, not an applicable matrix. Finalization sorts and
reduces contributions to produce canonical COO, CSR, or CSC. HypreCSR currently
uses process-local ownership on `hypre_MPI_COMM_SELF`, not distributed ownership.

Include `gendil/Algebra/SparseMatrixTypes/sparsematrixtypes.hpp` for all formats,
or the relevant header above for one format.

## Applying a matrix

Replacement and additive application use the same interface across formats:

```cpp
Apply(matrix, x, y);    // y = A*x, equivalent to matrix(x, y).
ApplyAdd(matrix, x, y); // y += A*x; here y becomes (-2,36).
```

Size `x` to `GetInputSize(matrix)` and `y` to `GetOutputSize(matrix)`.
These are scalar-vector dimensions, including for block matrices. Additive
application requires initialized output values. Use non-overlapping input and
output storage; in-place application is not a general sparse-matrix guarantee.
For a matrix with no stored entries, replacement zeros the output and addition
leaves it unchanged.

GenDiL vectors and enabled MFEM vectors are supported when their accessors match
the backend's memory space; input and output need not use the same vector class.
Applicable matrices satisfy [VectorMap](../vectormapconcept.hpp) for their
supported vector types. RawCOO buffers and matrix storage views do not.

### Selecting execution

A backend chooses how and where the matrix-vector product runs. Each applicable
matrix stores a backend; an explicit argument overrides it for one call:

```cpp
HostCSRBackend<> backend;
Apply(backend, matrix, x, y);    // Run this CSR product on the host.
ApplyAdd(backend, matrix, x, y); // Add another host product.
```

CPU defaults are host backends. CUDA/HIP builds default COO, CSR, and CSC to
vendor device backends. In device builds, BSR defaults to a vendor backend only
when generic vendor BSR support is available; otherwise it uses
`NativeDeviceBSRBackend`. Host and native-device alternatives remain selectable
where supported.

Assembly makes an additional BSR/SGBSR choice based on block shape and vendor
capabilities. HypreCSR uses its configured Hypre host/device backend and can
also forward explicitly to a CSR backend. See
[matrix assembly](../../FiniteElementMethod/MatrixAssembly/README.md) and
[vendor backends](VendorSparse/README.md) for supported configurations.

## Accessing and updating storage

Matrices own their allocations and release them automatically. BSR, COO, CSR,
CSC, RawCOO buffers, and RawCOO assembly layouts are move-only; transfer ownership
with `std::move`. SGBSR owns its BSR matrix and HypreCSR owns its CSR matrix.

For BSR, COO, CSR, CSC, and RawCOO buffers, acquire a synchronized view to inspect
or update storage. SGBSR and HypreCSR expose their contained BSR/CSR matrices:

| Access | Host | Device | Effect |
| --- | --- | --- | --- |
| Read | `GetHostReadView` | `GetDeviceReadView` | Synchronize to the selected side; expose read-only entries |
| Read/write | `GetHostReadWriteView` | `GetDeviceReadWriteView` | Preserve contents; make the selected side authoritative |
| Write | `GetHostWriteView` | `GetDeviceWriteView` | Skip synchronization; make the selected side authoritative |

Write access is appropriate for filling all entries, as in the first example.
Use read/write access when retaining existing contents. BSR, COO, CSR, and CSC
also offer values-only views, which keep structural indices read-only:

```cpp
auto coefficients = GetHostValuesReadWriteView(matrix);
coefficients.values[0] = 5; // Change A(0,0), preserving the sparsity pattern.
```

`GetKernelReadView<OnDevice>` and its read/write and write counterparts select
the memory side for full-array access. BSR additionally offers kernel
values-only wrappers. Sparse storage views contain raw pointers and are trivially
copyable for kernel capture. Keep their owner and referenced allocations valid
until access finishes. Const element/index template arguments determine
read-only access; a const view handle alone does not make its pointees const.

Synchronization is tracked per array. `Sync(matrix)` makes initialized arrays
current on both sides. Public metadata and `SyncHostDeviceArray` fields are
low-level access points: direct array access must first acquire `ReadHost`,
`ReadWriteHost`, `WriteHost`, or the device equivalent. Untracked writes are
unsupported; stale copies are never merged.

### Reusing vendor execution plans

Vendor backends cache execution plans for repeated products. Updating values
in the same allocation preserves a plan. Before changing dimensions, sparsity
indices, block structure, or allocations, reset the stored backend and every
explicit vendor backend used with that matrix. Mutable storage views do not
reset plans automatically. Cached backend instances must not be used
concurrently. See the [vendor cache contract](VendorSparse/README.md) for reset
calls and lifetime requirements.

## How SGBSR connects local blocks to field vectors

Finite-element coefficients may be shared between elements or related by
weighted restrictions. SGBSR applies an operator in three steps:

1. Gather the input field into the local coefficients expected by BSR.
2. Apply the BSR matrix.
3. Scatter local results back into the output field, accumulating shared entries.

For finite-element restriction maps, this represents
`E_test^* A_local E_trial`: the stored BSR coefficients do not include algebraic
restriction weights; gather and adjoint scatter apply those weights. This
differs from globally assembled COO/CSR/CSC, where those contributions have
already been combined.

`SGBSRMatrix` stores `bsr_matrix`, `trial_gather`, `test_scatter`, and reusable
`x_bsr`/`y_bsr` workspaces. One instance must not be applied concurrently.
The selected BSR backend also selects the gather/scatter memory space: host
execution uses OpenMP when enabled, while device execution keeps all three
stages on the device. Borrowed restriction indices and weights/signs must be
accessible there; memory validation does not transfer those maps.

Built-in restriction maps use atomic accumulation for potentially conflicting
scatter contributions, including shared H1 DoFs. Parallel results are expected
to agree within tolerance, not necessarily bit for bit. For supported domains
and restriction representations, see the
[assembly documentation](../../FiniteElementMethod/MatrixAssembly/README.md).

The [migration guide](../../../../scripts/migration/README.md) describes changes
from earlier sparse interfaces.
