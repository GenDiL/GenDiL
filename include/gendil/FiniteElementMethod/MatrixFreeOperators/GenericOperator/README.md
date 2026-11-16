# Generic weak-form execution

The generic operator evaluates a weak-form DSL expression using its named
field bindings, integration geometry, and quadrature rule. This guide covers
the numerical entry points, optional validation, and extension contracts for
custom facet connectivity and contexts. For mathematical expressions,
coefficient inputs, and complete examples, start with the
[weak-form DSL guide](../../WeakForm/README.md).

## Constructing and applying an operator

Given a form, context, and quadrature rule from that guide:

```cpp
auto op = MakeGenericOperator<SerialKernelConfiguration>(form, context, rules);
Vector x(op.InputSize()), y(op.OutputSize());
x = 1.0;
op(x, y); // Replace y with the evaluated residual.
```

Use `op(y)` for an input-free form, such as a load. The kernel policy selects
execution; GPU policies require a CUDA or HIP build, device-accessible data,
and device-callable coefficients. Keep referenced meshes, spaces, and
prescribed data alive while using an operator.

| API | Behavior |
| --- | --- |
| `MakeGenericOperator<Policy>(form, context, rules)` | Construct a matrix-free residual operator. |
| `MakeWeakFormResidualOperator<Policy>(form, context, rules)` | Equivalent residual-oriented factory name. |
| `MakePullbackGenericOperator<Policy>(form, context, rules)` | Equivalent underlying factory. |
| `op.InputSize()`, `op.OutputSize()` | Required algebraic vector extents. |
| `op(x, y)` | Evaluate a form with active inputs, replacing the entire output. |
| `op(y)` | Evaluate an input-free form such as a load, replacing the entire output. |
| `op.ApplyAdd(x, y)`, `op.ApplyAdd(y)` | Add into an existing output vector. |
| `GenericExplicitOperator<Policy>(form, context, rules, x, y)` | Direct additive evaluation; omit `x` for an input-free form. |
| `GenericAssembly<Format, Policy>(form, context, rules)` | Assemble a supported linear form into a sparse matrix. |
| `ValidateOperator(op, checks)` | Run explicitly selected host checks on a generic operator. |
| `ValidateOperator(form, context, rules, checks)` | Run the same checks before direct evaluation or assembly. |

`GenericExplicitOperator` adds to the existing output, unlike `op(x, y)`;
initialize that output before direct evaluation. For sparse assembly formats
and their restrictions, see the [assembly guide](../../MatrixAssembly/README.md).

## Optional quadrature validation

Local interior-facet execution requires corresponding tangential quadrature
rules to agree after the face's coordinate permutation or reflection. For
example, rotating an x-face onto a y-face requires matching rules on the
exchanged axes. Equal Gauss or Gauss-Lobatto rules satisfy this condition;
anisotropic rules remain valid when the actual orientations preserve them.
This compatibility is a precondition of local traversal. Use compatible rules
or global traversal when the condition does not hold.

To check local facet quadrature compatibility before using an operator, enable
the check in a policy defined at namespace scope:

```cpp
struct Checks : OperatorChecks
{
   static constexpr bool local_facet_quadrature = true;
};

auto op = MakeGenericOperator<SerialKernelConfiguration>(form, context, rules);
Checks checks;
ValidateOperator(op, checks);
op(x, y);

// Alternatively, validate before direct evaluation or sparse assembly:
ValidateOperator(form, context, rules, checks);
```

`OperatorChecks` disables all optional checks by default. The concrete policy
type selects checks at compile time; disabled checks are not instantiated, and
the `checks` object need not be `constexpr`. Both validation overloads return
`void`. A failed check prints a diagnostic and aborts, including in Release
builds. The operator overload supports the three generic factories listed above.

Construction, evaluation, and assembly do not automatically run these checks.
Validation is a separate host operation, including for GPU operators, and does
not launch a device kernel. It needs host-readable connectivity if inspection
is required. Rules and coordinate-orientation types that prove compatibility
avoid a mesh scan; otherwise validation examines actual local connectivity.
Global-facet and cell-only forms need no local quadrature check.

Validation stores no state in the operator. Repeat it after changing topology,
connectivity, or quadrature; changing only coefficient or field values does not
require revalidation. Mandatory DSL/context compatibility and vector-size
checks remain part of the numerical API.

## Implementing facet execution

These contracts are for extending connectivity, point factories, or kernels.
Weak-form authors use `Normal{}`, `minus`, `plus`, `jump`, and `average`;
the evaluator presents their shared mathematical convention through the
storage and accessors below.

### Stored sides and expression traces

Local face information exposes stored endpoints through `CurrentSide()` and
`NeighborSide()`. These identities remain fixed during local execution.
`GetNormalOrientation()` supplies the sign $s$ in
$n_F = s\,n_{\mathrm{current}}$, where $n_{\mathrm{current}}$ is the current
cell's outward unit normal and $n_F$ is the shared normal used by the form.
Thus `+1` means the normals agree and `-1` means they oppose; the sign does
not rename or reorder the stored sides. A local record's `GetReferenceNormal()`
remains the current cell's outward reference normal.

The weak-form evaluator uses this sign to select traces from local storage:

| Normal orientation | `minus(e)` reads | `plus(e)` reads |
| --- | --- | --- |
| `+1` | Current cell | Neighbor cell |
| `-1` | Neighbor cell | Current cell |

Local interpolation, geometry, and output ownership use current/neighbor.
Once a local face has a compile-time normal orientation, its `MinusSide()`
and `PlusSide()` accessors select the canonical endpoints in the table above.
Their `minus_side_type` and `plus_side_type` aliases follow that selection.
Runtime-oriented local records provide only current/neighbor accessors until
dispatch. Global face records already store canonical minus/plus endpoints
and retain their `MinusSide()`/`PlusSide()` interface.
The canonical local ordering describes interior facets; physical boundary
evaluation uses `CurrentSide()` and its outward `GetReferenceNormal()`.

Canonical endpoint selection does not transform reference coordinates or
Jacobian representations. `LocalFacetContext` applies the trace convention
to point data: it selects the two traces and normalizes coefficient Jacobian
inputs when needed. The local point factory applies the normal orientation
when constructing point data. Physical-gradient mapping continues to use each
stored side's unmodified inverse Jacobian.

This distinction between stored sides and expression traces preserves the
same form under local and global traversal. Local output always belongs to
the current cell. The default sign is static `+1`, appropriate for global
faces; custom local factories must supply the opposite sign where needed.
This sign is separate from `GetOrientation()`, which maps reference
coordinates and tensor indices. Built-in Cartesian faces encode normal
signs at compile time; unstructured faces can store them.

### Custom face information

Local execution passes a face with a compile-time normal orientation to its
interpolation and geometry helpers. Runtime signs are dispatched before those
helpers run; the stored connectivity is unchanged. Custom runtime face-info
types provide `CurrentSide()` / `NeighborSide()` and matching
`current_side_type` / `neighbor_side_type` aliases. They must also provide
`WithStaticNormalOrientation<Sign>(face)` in their own namespace.
It must preserve their endpoints and metadata and return a face
whose `GetNormalOrientation()` exposes the requested static sign. Already-static
custom faces pass through without rebinding.

### Point data and facet accessors

Custom local kernels obtain a `LocalFacetContext` from
`MakeLocalInteriorFacetContext(...)`, including when the form needs no neighbor
geometry. Local point data keeps current/neighbor geometry; access its
canonical traces with `GetMinusSide(facet, point)` and `GetPlusSide(facet, point)`.
For built-in contexts, side access requires an interior pair: physical boundaries
and already selected points cannot be selected again. Zero-argument connectivity
and storage accessors such as `MinusSide()` and `PlusSide()` keep their existing
roles. Selecting a trace leaves the stored normal unchanged.
Read their shared unit normal with `GetFacetNormal(facet, point)`. Built-in
point data stores the ready-to-use `physical_normal`: the local interior factory
applies the compile-time sign once, global interior data stores the canonical
normal, and boundary data stores the adjacent cell's outward normal. The accessor
returns this stored value, so repeated reads, untraced `n`, and both interior
traces use the same normal. Custom factories and direct point-data construction
must supply a canonical-minus outward unit normal on interior facets and an
outward unit normal on physical boundaries.
Point data without `physical_normal` requires a custom free `GetFacetNormal`
overload returning a unit normal with this same orientation convention. The
accessor does not compute a normal from `inv_J_mesh` and a reference normal.
`ComputePhysicalNormal` remains available for explicit geometry calculations;
its result is unnormalized.

Built-in point data and selected traces own their values and retain no reference
to the facet; selecting from a temporary facet is safe. The stored normal does
not need the originating facet to interpret its orientation. Keep the
corresponding local facet context for trace selection and coefficient-Jacobian
normalization. The facet argument remains part of `GetFacetNormal` for custom
overloads.

`Normal{}`, `minus`, `plus`, and `upwind` call the normal and side accessors
without qualification, allowing custom contexts to provide overloads in their
own namespace. The overloads must preserve canonical trace ordering and the
shared interior or outward boundary unit-normal convention.
Custom side accessors retain their own reference or value return semantics.
Built-in overloads retain global and boundary evaluation's existing contexts;
no additional context wrapper is required for normal access.

### Coefficient Jacobian inputs

`JacobianInverse` coefficient inputs call
`GetCoefficientJacobianInverse(context, point)` without qualification, so custom
contexts can provide their own overloads. On local facets this expresses the
selected inverse Jacobian in the canonical-minus reference frame, obtaining
the reference-coordinate orientation only when needed. Physical gradients keep
using the raw matrix. Built-in cell, global-facet, and boundary inputs read
`point.inv_J_mesh` directly.

Custom implementations of `GetMinusSide`, `GetPlusSide`, and
`GetCoefficientJacobianInverse` use free overloads in the context or point type's
namespace. Mark overloads used in GPU kernels `GENDIL_HOST_DEVICE`.
The former member hooks `facet.MinusSide(point)`, `facet.PlusSide(point)`,
`context.GetCoefficientJacobianInverse(raw)`, and
`point.GetCoefficientJacobianInverse()` are no longer called; convert these
customizations to the corresponding free overloads.

`GetCoefficientJacobianInverse` preserves a const reference when it can use a
named built-in point's matrix directly. Transformed results and results from
temporary built-in points own their storage. A returned reference remains valid
only while its point data remains alive.
Custom endpoint or orientation getters may return values: their temporaries
remain alive through normalization, without cached orientation storage or a
borrowed reference in the selected point. Neighbor cell geometry is
acquired only when the form requires it.

## Source map

- [Operator construction and invocation](genericoperator.hpp).
- [Facet contexts and canonical local trace access](Context/facetcontext.hpp).
- [Global trace access](Context/facetsides.hpp),
  [stored normals](Context/facetnormal.hpp), and
  [coefficient Jacobian access](Context/coefficientjacobian.hpp).
- [Point-data construction](Context/quadraturepointcontext.hpp).
- [Local facet execution](LocalFacetOperator/localinteriorfacet.hpp) and
  [global facet execution](GlobalFacetOperator).
- [Face information and orientation dispatch](../../../Meshes/Connectivities/faceconnectivity.hpp).
- [Opt-in operator validation](../../OperatorValidation/operatorvalidation.hpp).
