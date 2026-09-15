# GenDiL views

Use a view to access data with vector, matrix, or tensor indices, without
computing a flat-buffer position at every access. For example, interpret six
existing values as a 2-by-3 tensor:

```cpp
#include <gendil/Utilities/View/Layouts/stridedlayout.hpp>
using namespace gendil;

Real values[6]{};
auto tensor = MakeFIFOView(values, 2, 3);
tensor(1, 2) = 5; // Writes values[5]; no data is copied.
```

`MakeFIFOView` uses first-coordinate-fastest storage: `(0,j)` and `(1,j)` are
adjacent, and `(i,j)` accesses `values[i + 2*j]`. Keep the buffer alive and
unmoved while using the view. Subsequent examples also use names from `gendil`.

## How a view selects and accesses an entry

`View<Container, Layout>` combines two responsibilities:

- The layout maps coordinates to a container key: here, the position `i + 2*j`.
- The container determines what access means: here, a reference to that value.

`MakeView(data, layout)` lets you choose the layout explicitly. A stride is the
increment in container entries when one coordinate increases by one. Strides
`{1,2}` give the indexing above; other strides can describe padded storage.
See [strided layouts](Layouts/stridedlayout.hpp) for construction options.

The container and layout are stored by value. This need not copy field values:
`PointerContainer` and `ReferenceContainer` borrow storage, whereas
`StaticContainer` owns inline values. Not every view is non-owning.

## Querying dimensions when you need them

An extent is an axis length; rank is the number of axes. Indexing needs strides,
not necessarily lengths: `MakeFIFOView` does not retain its sizes. If a consumer
also needs dimension or entry-count queries, supply shape at construction:

```cpp
Real shaped_values[6]{};
auto layout = StridedLayout<2, DynamicExtents<2>>{
   std::array{1, 2}, DynamicExtents<2>{2, 3}};
auto shaped = MakeView(shaped_values, layout);
auto first_dimension = GetExtentValue<0>(shaped); // 2
auto entry_count = shaped.Size();               // 6
```

Choose `StaticExtents<Sizes...>` for compile-time lengths, `DynamicExtents<Rank>`
for runtime lengths, or `NoExtents<Rank>` when lengths need not be available.
Unknown does not mean zero. There is no attachment or refinement API.

`GetExtentValue<Axis>` returns an ordinary integer; `GetExtent<Axis>` preserves
static-axis compile-time information. `HasExtent` and `HasCompleteExtents`
describe query availability. `Size()` and `GetLogicalSize` require a complete
shape and return a `GlobalIndex`, usable in constant expressions when possible.

| Layout | Available dimensions |
| --- | --- |
| `VectorLayout` | One runtime length |
| `FixedStridedLayout` | Intrinsic static tensor dimensions |
| `StridedLayout<Rank, Extents>` | None by default; optional complete static or dynamic shape |
| Direct and weighted restriction layouts | Static local DoF axes; unknown element axis |
| `OffsetLayout` | Its parent's available dimensions |
| `OrientedLayout<Rank>` | None; construction sizes are not retained |

Logical size is not allocation size, algebraic DoF extent, or sparse capacity.
Neither shape nor strides alone guarantee contiguous scalar storage. See
[extents](extents.hpp) for query and representation details.

## Working with part of a vector

`MakeVectorView(parent, offset, extent)` selects a contiguous range without
copying values. For a GenDiL `Vector`, synchronization stays with the parent:

```cpp
#include <gendil/Algebra/vectoraccess.hpp>

Vector field(6);
auto initial = WriteKernelVector<false>(field); // Host access; initialize all entries.
for (size_t i = 0; i < initial.Size(); ++i) { initial[i] = Real(i); }

auto block = MakeVectorView(field, 2, 3);
block[1] = 9; // Updates field[3]; block.Size() is 3.
```

Construction performs no synchronization. Nested subranges combine offsets and
retain the root parent, not an intermediate view. Keep the parent alive and
unmoved. `MakeVectorView(pointer, extent)` instead views already-selected raw
storage, with no owner to synchronize.

### Kernel access

Acquire kernel access before capture to select host or device storage and get
a small pointer-backed view. Continuing with `block` above:

```cpp
#include <gendil/Utilities/Loop/kernelloop.hpp>

constexpr bool OnDevice = false; // Host execution for this example.
auto output = ReadWriteKernelVector<OnDevice>(block);
KernelLoop<OnDevice>(output.Size(), [=] GENDIL_HOST_DEVICE (size_t i) mutable
{
   output[i] *= 2;
}); // field[2..4] now contains 4, 18, 8.
```

Use `ReadKernelVector` for reads, `ReadWriteKernelVector` to preserve and modify
values, and `WriteKernelVector` when old values need not be preserved.
Synchronization and validity apply to the whole owner. Write-only access does
not synchronize untouched entries; use read-write access when they must survive
a host/device transfer.

Writable pointer-backed captures require `mutable`; read-only kernel views
remain read-only in mutable closures. With a configured device backend, use
`OnDevice = true` and keep the selected storage alive and unmoved until execution
finishes. Capturing a reference to a host vector does not make it GPU-accessible.
Indirected access also needs its index map in the selected memory space.
Low-level host/device accessors still return pointers for backend interoperability.

## Constness and lifetime

The container decides const semantics. Const `PointerContainer` and
`StaticContainer` access is read-only. `ReferenceContainer` delegates to its
parent's const accessor, which is read-only for GenDiL `Vector`.
`HostDevicePointer<T>` permits mutation unless `T` is const.
`IndirectedContainer` preserves its child's reference, value, or proxy behavior.

Temporary pointer and reference handles may safely expose caller-owned storage.
Temporary inline-owning views cannot: references would outlive their values.
Weighted proxies also require a named view because they borrow its embedded
handles. `View` forwards access to the container, which decides temporary-access
safety.

## Specialized tensor and finite-element access

### Dimension slices and orientation

`MakeDimensionSubView<Axis>(view, coordinates...)` keeps one axis of a direct
fixed-layout tensor, preserving its static length and stride. Supply one
coordinate per axis; the selected axis's coordinate is ignored. For example,
`MakeDimensionSubView<1>(view, i, 0)` fixes axis zero at `i`. It supports
pointer-backed and named inline-owning views; const input gives a read-only
slice. The buffer must outlive the slice. Generic strided, oriented, indirected,
and weighted slicing is not provided. See the
[contraction helpers](../../FiniteElementMethod/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp).

`MakeOrientedView(data, sizes, orientation)` reorders or reverses axes.
Supply positive sizes in native axis order; they determine the mapping but are
not retained. Runtime `Permutation`, static identities, and their tensor products
are supported. DoF execution uses runtime permutations for nonidentity
orientations and a separate semantic path for weighted/global DoF access. See
[oriented layouts](Layouts/orientedlayout.hpp).

[ThreadedView](threadedview.hpp) distinguishes the collective tensor shape from
per-thread register storage, exposing separate extent types rather than one
`Size()`.

### Weighted finite-element values

A constrained local degree of freedom may depend on several algebraic entries.
`MakeWeightedRestrictionView` exposes it through a `WeightedDofReference` proxy:

- `Materialize(access)` gathers the weighted value.
- Assignment applies adjoint scatter-assign, not inverse interpolation.
- `operator+=` applies non-atomic adjoint scatter-add.
- `AtomicAddInPlace` applies atomic adjoint scatter-add.

Materialize before `auto` value deduction or unconstrained numeric-template
dispatch. Proxies cannot survive movement or destruction of their view; a helper
taking a view by value must not return a proxy borrowing its local copy. Gather
ordinary local values before contractions. Nodal interpolation still requires
one-entry, unit-weight, direct-reference restrictions. See
[weighted restriction views](../../FiniteElementMethod/Restrictions/weightedrestrictionview.hpp)
for scatter uniqueness, concurrency, constness, and lifetime requirements.

Generic consumers must distinguish readable values, direct references, and
scatter operations. `is_view_v<T>` identifies `View` types; it does not prove
those capabilities.

## Validity and checks

Supply valid dimensions, strides, coordinates, and subranges. Extent inputs are
nonnegative, non-Boolean integers; lengths, required products, and offsets must be
representable. Generic extent storage permits zero lengths; fixed and oriented
layouts require positive dimensions. Rank zero represents one scalar.

Static metadata constraints are checked at compile time. Runtime layout and
subrange checks run only in CPU debug builds; release and GPU execution rely on
host setup establishing validity. Indexing is unchecked. Layout headers detail
construction requirements, including intermediate-product bounds. Allocation,
mesh, and restriction validation remain separate from these view checks.
