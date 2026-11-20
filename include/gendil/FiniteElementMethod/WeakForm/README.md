# Weak-form DSL

GenDiL weak forms describe finite-element operators as integrals over cells
and facets. Build an expression from named fields, coefficients, and
mathematical operations, then bind its names to spaces, data, and a mesh.
The same linear form can be evaluated as a matrix-free operator or assembled
into a sparse matrix. Nonlinear residuals can also be evaluated matrix-free.

Include `<gendil/gendil.hpp>` and use the `gendil` namespace for the examples
below. The [main README](../../../../README.md#weak-form-dsl) gives a shorter
introduction with advection and diffusion examples.

- [A first operator](#a-first-operator)
- [Fields and integration domains](#fields-and-integration-domains)
- [Algebra and derivatives](#algebra-and-derivatives)
- [Coefficients](#coefficients)
- [Facet operations](#facet-operations)
- [SIPDG and inverse normal cell size](#sipdg-and-inverse-normal-cell-size)
- [Combining forms and extracting loads](#combining-forms-and-extracting-loads)
- [Evaluation and assembly](#evaluation-and-assembly)
- [Current scope](#current-scope)

## A first operator

This complete example describes the reaction-diffusion residual

$$
R(u;v)=\int_\Omega \nabla u\cdot\nabla v+uv-fv\,dx,
\qquad f(x,y)=1+x.
$$

It uses a continuous Cartesian finite-element space on the unit square,
with homogeneous natural boundary flux. `lhs` extracts the linear operator
and `rhs` extracts the load, so the discrete equation is $Ax=b$.
The example applies the operator and evaluates the load; solving the linear
system is a separate step.

```cpp
#include <gendil/gendil.hpp>
#include <array>
using namespace gendil;

int main()
{
   CartesianMesh<2> mesh({4, 4}, {0.25, 0.25}, {0.0, 0.0}, false);
   auto finite_element = MakeLobattoFiniteElement(FiniteElementOrders<1, 1>{});
   auto space = MakeFiniteElementSpace(
      mesh, finite_element, CartesianH1RestrictionSpecification{});
   auto rules = MakeIntegrationRule(IntegrationRuleNumPoints<3, 3>{});

   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   Cells<"mesh"> cells;
   auto f = MakeCoefficient<"source", PhysicalCoordinates>(
      [] GENDIL_HOST_DEVICE (const std::array<Real, 2>& X) -> Real
      { return 1.0 + X[0]; });

   auto residual = integrate(
      cells, dot(grad(u), grad(v)) + u * v - f * v);
   auto context = MakeWeakFormContext(
      MakeTrialField<"u">(space),
      MakeTestField<"v">(space),
      MakeIntegrationDomain<"mesh">(mesh));

   auto A = MakeGenericOperator<SerialKernelConfiguration>(
      lhs(residual), context, rules);
   auto load = MakeGenericOperator<SerialKernelConfiguration>(
      rhs(residual), context, rules);

   Vector x(A.InputSize()), y(A.OutputSize()), b(load.OutputSize());
   x = 1.0;
   A(x, y);
   load(b);
}
```

The remaining examples reuse names from this setup where indicated. A field
declaration does not choose a basis, a polynomial degree, or continuity:
those come from the space registered in the context.

## Fields and integration domains

Trial fields read the operator's active input; test fields identify where
its residual is accumulated. Prescribed fields read separately supplied
finite-element data.

| Declaration | Meaning |
| --- | --- |
| `TrialSpace<"u"> u;` | Named scalar trial field. |
| `TestSpace<"v"> v;` | Named scalar test field. |
| `VectorTrialSpace<"U"> U;` | Named vector trial field; equivalent to `TrialSpace<"U", FieldShape::Vector>`. |
| `VectorTestSpace<"V"> V;` | Named vector test field; equivalent to `TestSpace<"V", FieldShape::Vector>`. |
| `FiniteElementField<"w"> w;` | Prescribed field value, with shape inferred from its registered space. |

Use a vector finite-element space for vector symbols. The number of vector
components comes from that space, not from the symbol's name.

| Domain | Entities integrated |
| --- | --- |
| `Cells<"mesh"> cells;` | Cells of the named mesh or partition. |
| `InteriorFacets<"mesh"> faces;` | Interior facets, including supported nonconforming subfacets. |
| `BoundaryFacets<"mesh"> boundary;` | Physical boundary facets. |

`integrate(domain, expression)` associates an integrand with one of these
domains. Quadrature weights and the physical integration measure are applied
automatically. Each integrand must be scalar and linear in one named test
field. For example, `u * v`, `dot(grad(u), grad(v))`, and `f * v` qualify;
`u * v * v` does not. Trial dependence may be nonlinear.

The context connects these symbolic names to numerical objects:

| Context entry or factory | Purpose |
| --- | --- |
| `MakeTrialField<"u">(space)` | Bind a trial name to a finite-element space. |
| `MakeTestField<"v">(space)` | Bind a test name to a finite-element space. |
| `MakeFiniteElementField<"w">(space, dofs)` | Bind prescribed data and the space that interprets it. |
| `MakeIntegrationDomain<"mesh">(mesh_or_partition)` | Bind integration geometry and topology. Pass a mesh or partition, not a finite-element space. |
| `MakeWeakFormContext(entries...)` | Collect the named bindings used to construct an operator. |

The expressions determine trial and test roles; the two space-registration
factories have the same binding behavior. Trial and test symbols may share
a name when their shapes agree. An active trial name cannot also identify
prescribed data. Keep referenced meshes, spaces, and prescribed data alive
while using the operator.

Several trial or test names may occur in a form. They use one common input
vector and one common output vector, with placement determined by their
spaces and restrictions. Distinct names alone do not allocate separate
vector blocks. A form may use several integration domains, but cannot mix
mesh-bound and partition-bound domains.

See [field declarations](trialspace.hpp), [prescribed fields](finiteelementfield.hpp),
[domains](domain.hpp), and [context registration](weakformcontext.hpp).

## Algebra and derivatives

In the following table, `a` and `b` are scalar expressions, `x` and `y` are
vectors, and `A` and `B` are matrices. Shapes and dimensions must agree.
“Test-free” means that an expression contains no test field; it may still
depend on trial fields or prescribed data.

| Expression | Meaning | Restrictions in an integrated form |
| --- | --- | --- |
| `ScaleExpr{c}` | Scalar constant expression. | Useful when a constant is an operand of addition, e.g. `u + ScaleExpr{1.0}`. |
| `e + f`, `e - f`, `-e` | Addition, subtraction, negation. | Addition and subtraction require matching shapes. |
| `2.0 * e` | Scale an expression by a real constant. | Literal scaling is supported on the left. |
| `a * b` | Scalar product. | At most one operand may contain a test field. |
| `a * x`, `x * a`, `a * A`, `A * a` | Scalar multiplication of a vector or matrix. | If a test field occurs, the scalar multiplier must be test-free. |
| `A * x` | Matrix-vector product. | At most one operand may contain a test field. |
| `A * B` | Matrix-matrix product. | The columns of `A` must match the rows of `B`; at most one operand may contain a test field. |
| `dot(x, y)` | Vector dot product. | At most one operand may contain a test field. |
| `inner(x, y)`, `inner(A, B)` | Euclidean or Frobenius inner product. | Matching non-scalar shapes; at most one operand may contain a test field. |
| `outer(x, y)` | Matrix with entries $x_i y_j$. | If a test field occurs, it must be in the first operand. |
| `transpose(A)` | Matrix transpose. | Rectangular matrices and test-linear operands are supported. |
| `sym(A)`, `skew(A)` | $(A+A^T)/2$ and $(A-A^T)/2$. | Square matrix; test-dependent operands are supported. |

There are no DSL overloads for division, powers, or transcendental functions.
Use ordinary C++ arithmetic inside coefficient callbacks for such operations.
For scalar products use `a * b`; for vector products use `dot`, `inner`, or
`outer` according to the intended contraction.

Derivatives are physical derivatives, with the cell mapping accounted for.
Here `u` is a named scalar trial or test field, `U` a named vector trial or
test field, and `w` a prescribed vector finite-element field.

| Expression | Result and convention | Availability |
| --- | --- | --- |
| `grad(u)` | Physical gradient vector. | Cells and facets. |
| `grad(U)` | Matrix with entries $\partial_j U_i$ (component, derivative direction). | Cells and facets. |
| `normal_derivative(u)` | Scalar `dot(grad(u), Normal{})`. | Facets. |
| `normal_derivative(U)` | Vector `grad(U) * Normal{}`. | Facets. |
| `sym_grad(U)`, `skew_grad(U)` | Symmetric or skew part of `grad(U)`. | One vector component per spatial dimension. |
| `div(U)`, `div(w)` | Scalar $\sum_i \partial_i U_i$. | One vector component per spatial dimension. |
| `curl(U)`, `curl(w)` | Physical curl vector. | Three dimensions and three vector components. |

Apply these derivatives to named fields, then combine or trace their
results: for example, `grad(u) + grad(p)` or
`average(normal_derivative(u))`. To obtain the gradient of a prescribed field,
use the coefficient input `VectorFieldGradient<"w">` described below for a
vector field, or `FieldGradient<"s">` for a scalar field.

For example, a vector diffusion or elasticity contribution can use
`inner(grad(U), grad(V))`, `inner(sym_grad(U), sym_grad(V))`, or
`div(U) * div(V)`. The latter two require spatially matching vector dimensions.

Matrix products can act on either index of a vector-field gradient:
`A * grad(U)` mixes field components, while `grad(U) * K` acts on derivative
directions. With test-free matrix coefficients `A` and `K`, the following
integrands express the same contraction:

```cpp
auto trial_product = inner(A * grad(U) * K, grad(V));
auto test_product = inner(grad(U), transpose(A) * grad(V) * transpose(K));
```

Matrix products support rectangular shapes and nested products. Products with
two test-dependent operands remain nonlinear in the test field and are rejected.

Transpose may also act on a test expression: `inner(F, transpose(grad(V)))`
is equivalent to `inner(transpose(F), grad(V))` for a test-free matrix `F`
of compatible dimensions. On facets, transpose can be placed inside or
outside `jump` and `average`.

See [products](productexpr.hpp), [gradients](gradient.hpp),
[normal derivatives](normalderivative.hpp), [symmetric and skew parts](symmetry.hpp),
[divergence](divergence.hpp), and [curl](curl.hpp).

## Coefficients

A coefficient evaluates a user-supplied function at a quadrature point.
Its input tags state which coordinates, geometry, or field data the callback
needs. Arguments arrive in the order of the tags; an empty input list gives
a constant callback.

| Factory | Declared result shape | Typical callback result |
| --- | --- | --- |
| `MakeCoefficient<"name", Inputs...>(fn)` | Scalar | `Real` |
| `MakeVectorCoefficient<"name", Inputs...>(fn)` | Vector | `std::array<Real, d>` |
| `MakeMatrixCoefficient<"name", Inputs...>(fn)` | Matrix | `std::array<std::array<Real, d>, d>` |

Vector and matrix values may also use custom types satisfying `StaticVector`
or `StaticMatrix`. Shared arithmetic reads entries through `GetValue` and
the declared scalar type, so inputs can be read-only views or return scalar
proxies. The types supply static dimensions and readable entries as described
in the [static algebra concepts](../../Algebra/staticalgebraconcepts.hpp).
The generic operations do not require writable inputs or default constructors.

Arithmetic results own their entries, using this storage policy:

| Operation and operands | Result storage |
| --- | --- |
| Addition/subtraction of two standard-array vectors or matrices | Matching `std::array` storage. |
| Negation fallback for a standard-array vector or matrix | Matching `std::array` storage. |
| Scalar multiplication of a standard-array vector | `std::array`. |
| Scalar multiplication of a standard-array matrix | `SerialRecursiveArray`. |
| Matrix-vector or matrix-matrix multiplication | `SerialRecursiveArray`, including for standard-array operands. |
| Generic arithmetic on mixed storage or custom readable types | `SerialRecursiveArray`. |
| Native recursive-array operations | Existing storage and scalar-type behavior. |

Generic entry types follow scalar arithmetic promotion. `NegateValue` honors
an existing native unary minus before applying its vector/matrix fallback.
Operators must be visible where expressions are defined; outside `gendil`,
bring the desired operators into scope. C++ does not permit these arithmetic
overloads for raw-array-only operands, although a compatible raw vector can
be paired with a class-type vector.

The coefficient name labels the expression; it does not require a context
entry. Names referenced by `FieldValue` and `FieldGradient` do require field
bindings. For example, given `conductivity_space` and `conductivity_dofs`:

```cpp
auto k = MakeCoefficient<"diffusivity", FieldValue<"conductivity">>(
   [] GENDIL_HOST_DEVICE (Real value) -> Real { return value; });
auto material_context = MakeWeakFormContext(
   MakeTrialField<"u">(space),
   MakeTestField<"v">(space),
   MakeFiniteElementField<"conductivity">(
      conductivity_space, conductivity_dofs),
   MakeIntegrationDomain<"mesh">(mesh));
auto diffusion = integrate(cells, k * dot(grad(u), grad(v)));
```

For prescribed vector values used in `dot`, `inner`, or `outer`, expose
`VectorFieldValue<"w">` through `MakeVectorCoefficient` with an identity callback.
This supplies the explicit result shape those operations require before
context binding; a bare `FiniteElementField<"w">` has an unresolved shape
at that stage.

These are all the coefficient input tags. “Selected side” on an interior
facet means evaluation under `minus`, `plus`, or an operation such as
`average` that evaluates both sides.

| Input tag | Value supplied | Cells | Boundary facets | Interior facets |
| --- | --- | --- | --- | --- |
| `ElementIndex` | Integration element index. | Current cell | Adjacent cell | Current cell locally; canonical-minus cell globally, even under `plus`. |
| `QuadraturePointIndex` | Quadrature multi-index. | Yes | Yes | Traversal quadrature index. |
| `PhysicalCoordinates` | Physical evaluation point. | Yes | Yes | Shared physical point. |
| `ReferenceCoordinates` | Parent reference-cell coordinates. | Yes | Yes | Not supported as correct side/facet coordinates; see below. |
| `Jacobian` | Parent-cell mapping Jacobian. | Yes | Yes | Not currently supplied. |
| `JacobianDeterminant` | Parent-cell Jacobian determinant, not the facet integration measure. | Yes | Yes | Not currently supplied. |
| `JacobianInverse` | Inverse parent-cell Jacobian. | Yes | Yes | Selected side, expressed in the canonical-minus reference-axis frame. |
| `InverseNormalCellSize` | $\lVert J_K^{-T}\widehat n\rVert$. | No | Adjacent cell | Selected side. |
| `FieldValue<"u">` | Scalar field value (default shape). | Yes | Yes | Selected side. |
| `FieldGradient<"u">` | Scalar field's physical gradient (default field shape). | Yes | Yes | Selected side. |
| `FieldValue<"u", FieldShape::Scalar>` | Explicitly scalar field value. | Yes | Yes | Selected side. |
| `FieldGradient<"u", FieldShape::Scalar>` | Explicitly scalar field's physical gradient. | Yes | Yes | Selected side. |
| `VectorFieldValue<"w">` | Explicitly vector field value. | Yes | Yes | Selected side. |
| `VectorFieldGradient<"w">` | Explicitly vector field's physical gradient. | Yes | Yes | Selected side. |

Interior field and side-geometry inputs have no implicit minus-side default.
Select a trace of the coefficient or an enclosing expression, for example
`minus(k)`, `plus(k)`, or `average(k)`. Operations such as `jump` and the
transported operand of interior `upwind` also establish the selected sides.

`FieldValue<Name, Shape>` and `FieldGradient<Name, Shape>` declare a field
shape, with `Shape` defaulting to `FieldShape::Scalar`.
`VectorFieldValue<Name>` and `VectorFieldGradient<Name>` are aliases for
`FieldValue<Name, FieldShape::Vector>` and
`FieldGradient<Name, FieldShape::Vector>`. All declarations are checked
against the selected FE space during operator construction, including the
selected interior-facet side, even if a generic callback accepts any rank.
These spellings work for both active trial and prescribed finite-element fields.

The shape declaration describes the **underlying field**. For a scalar field
on a mesh of physical dimension `d`, the gradient has length `d`. For a field
with `m` components, the gradient has shape `m × d`, indexed as
`dw(component, direction)`. The bound space supplies these extents; `m` need
not equal `d`. Index, geometry, and field inputs use representations determined
by the mesh and space; generic callback parameters (`const auto&`) are useful
for those inputs.

The factory independently declares the callback's **output** shape. For example,
this scalar coefficient reads a three-component field on a 2D mesh, with
arguments supplied in the order of the input tags:

```cpp
auto c = MakeCoefficient<"c", VectorFieldValue<"w">, VectorFieldGradient<"w">>(
   [] GENDIL_HOST_DEVICE (const auto& w, const auto& dw) {
      return w(2) + dw(2, 1);
   });
```

For compile-time inspection, all field input tags expose `field_shape` and
`argument_rank`: scalar value/gradient ranks are 0/1, and vector value/gradient
ranks are 1/2.
Declarations support scalar and componentwise vector FE fields.

The interior `ReferenceCoordinates` provider currently falls back to a
reference integration-rule point; it does not map that point to the selected
cell's facet. Do not use it for side-dependent reference-coordinate formulas.
Similarly, selecting `plus` does not change the `ElementIndex` callback input.
`ElementIndex` and `QuadraturePointIndex` describe the traversal: local and
global visits need not supply the same indices at a physical point. Together
with interior `ReferenceCoordinates`, they are outside the local/global
operator-equivalence guarantee. Use physical coordinates or traced field
values for coefficients that must be independent of traversal.

Callbacks may read active trial fields through `FieldValue` or `FieldGradient`
when those fields are declared in the form, enabling nonlinear material laws.
Such opaque callback dependence cannot be analyzed by `lhs` and `rhs`.
Callbacks are stored by value and invoked as const callables. For device
execution, mark them `GENDIL_HOST_DEVICE` and use device-accessible captures
and operations.

See [coefficient factories and providers](coefficient.hpp) and
[input definitions](coefficientinputs.hpp).

## Facet operations

In weak-form expressions on an interior facet, `minus(e)` and `plus(e)`
identify the same two cells regardless of which cell is being processed.
The mesh connectivity determines this fixed side ordering.
The physical unit normal `n = Normal{}`
points outward from the chosen minus cell toward the plus cell. Both traces
use that same normal: `plus(Normal{})` does not reverse it. On a boundary,
the normal points out of the domain.

Local traversal visits the facet from each cell; global traversal visits it
once and contributes to both test rows. With compatible quadrature, these
execution patterns give the same operator, including for one-sided
coefficients such as `minus(k) * jump(u) * jump(v)`.

| Expression | Interior facets | Boundary facets |
| --- | --- | --- |
| `Normal{}` | Unit normal oriented from minus to plus. | Outward unit normal. |
| `minus(e)`, `plus(e)` | Evaluate `e` with the selected side's fields and available geometry. | Unsupported. |
| `jump(e)` | `minus(e) - plus(e)`; no implicit multiplication by a normal. | `e`. |
| `average(e)` | `(minus(e) + plus(e)) / 2`. | `e`. |
| `normal_projection(x)` | Vector $(x\cdot n)n$. | Same definition. |
| `tangential_projection(x)` | Vector $x-(x\cdot n)n$. | Same definition. |
| `upwind(beta, e)` | Normal flux $(\beta\cdot n)e^-$ for nonnegative speed, $(\beta\cdot n)e^+$ otherwise. | Unsupported; supply inflow data. |
| `upwind(beta, u, g)` | Unsupported; use the two-argument form. | Normal flux $(\beta\cdot n)u$ on outflow and $(\beta\cdot n)g$ on inflow. |

Vector projections require as many components as the physical dimension.
In particular, `normal_projection(x)` is a vector, whereas `dot(x, Normal{})`
is a scalar normal component. Jumps and averages act componentwise.
For coefficients returning vectors or matrices, both `average` and `jump`
use the shared arithmetic for `StaticVector` and `StaticMatrix` values,
including standard arrays, recursive arrays, and custom readable types.
Fractional scaling and averaging retain fractional results for integer
entries, and averaging preserves floating-point precision.

Both local and global facet forms support `jump(v)`, `average(v)`, and
explicit test traces such as `minus(v)` and `plus(v)`, including traced
test gradients. Do not nest trace selectors.

The [generic execution guide](../MatrixFreeOperators/GenericOperator/README.md#implementing-facet-execution)
explains how custom connectivity and kernels implement these trace conventions.

`upwind` returns the normal **flux**, including the speed, rather than just
the selected state. Its velocity must be a test-free spatial vector and its
transported expression must be test-free. The two-argument interior overload
supports direct `grad(u)` as well as coefficient callbacks with `FieldGradient`
inputs. For example, `upwind(beta, grad(u))` returns the normal speed times
the selected side's **physical gradient**, using that cell's inverse Jacobian.
The result is a vector for scalar `u` or a matrix for vector `u`; contract it
with a compatible test expression using `dot` or `inner` to form a scalar
integrand. For discontinuous velocities, choose a shared velocity explicitly,
for example `average(beta)`, if that is the numerical flux required by the model.
Zero normal speed gives zero flux.

For example, the following advection form includes prescribed inflow. Unlike
the first example, it binds the trial and test fields to a broken space:

```cpp
auto dg_space = MakeFiniteElementSpace(mesh, finite_element);
auto dg_context = MakeWeakFormContext(
   MakeTrialField<"u">(dg_space),
   MakeTestField<"v">(dg_space),
   MakeIntegrationDomain<"mesh">(mesh));
InteriorFacets<"mesh"> faces;
BoundaryFacets<"mesh"> boundary;
auto beta = MakeVectorCoefficient<"velocity">(
   [] GENDIL_HOST_DEVICE () { return std::array<Real, 2>{1.0, 0.5}; });
auto inflow = MakeCoefficient<"inflow">(
   [] GENDIL_HOST_DEVICE () -> Real { return 1.0; });

auto advection =
   integrate(cells, -u * dot(beta, grad(v)))
   + integrate(faces, upwind(beta, u) * jump(v))
   + integrate(boundary, upwind(beta, u, inflow) * v);
```

See [traces](trace.hpp), [jumps](jump.hpp), [averages](average.hpp),
[projections](facetprojection.hpp), and [upwinding](upwind.hpp).

## SIPDG and inverse normal cell size

`InverseNormalCellSize` supplies the inverse cell scale normal to a facet:

$$
\rho_K(q)=\|J_K(q)^{-T}\widehat n\|.
$$

Here $J_K$ is the cell mapping's Jacobian and $\widehat n$ is the unit facet
normal on GenDiL's reference cell $[0,1]^d$. For a Cartesian cell, this is the
inverse of its width normal to the facet. On curved cells it is a local scale
at each facet quadrature point. It may differ between adjacent cells and is
not the inverse length of their shared facet or subfacet.

Interior uses must evaluate the coefficient in a side-selected context,
for example with `minus`, `plus`, or `average` around it or an expression
containing it. Boundary uses take the sole adjacent cell's value. A shared
penalty can be built from the two cell contributions. For unit conductivity
and homogeneous Dirichlet boundary data, the complete SIPDG form is:

```cpp
constexpr int p = 1; // Matches the finite element in the examples above.
auto tau = MakeCoefficient<"penalty", InverseNormalCellSize>(
   [] GENDIL_HOST_DEVICE (Real inverse_normal_size) -> Real
   { return 10.0 * (p + 1) * (p + 1) * inverse_normal_size; });

auto sipdg =
   integrate(cells, dot(grad(u), grad(v)))
   + integrate(
        faces,
        -average(normal_derivative(u)) * jump(v)
        - jump(u) * average(normal_derivative(v))
        + average(tau) * jump(u) * jump(v))
   + integrate(
        boundary,
        -normal_derivative(u) * v - u * normal_derivative(v)
        + tau * u * v);

auto sipdg_operator = MakeGenericOperator<SerialKernelConfiguration>(
   sipdg, dg_context, rules);
```

The factor proportional to $(p+1)^2$ is illustrative, not a universal
stability bound. Choose a sufficiently large factor for the discretization,
material, and mesh. The geometric input alone does not guarantee stability.

Each side's callback is evaluated before averaging: `average(tau)` means
$\tfrac12(\tau(\rho^-)+\tau(\rho^+))$, including for nonlinear callbacks.
It does not apply the callback to an averaged input. For varying scalar
conductivity `k`, use `average(tau * k)` on interiors and `tau * k` on
boundaries, with matching conductivity-weighted consistency terms. The
[main README](../../../../README.md#weak-form-dsl) shows that form. Nonzero
Dirichlet values also require the corresponding boundary load terms.

## Combining forms and extracting loads

| Operation | Meaning |
| --- | --- |
| `integrate(domain, e)` | One scalar contribution, linear in one named test field. |
| `form1 + form2` | Sum of integral contributions. |
| `lhs(R)` | Linear trial-dependent part of a jointly affine residual. |
| `rhs(R)` | Negative of the trial-independent part of that residual. |

Put subtraction and scaling inside the integrand; general `form1 - form2`
and `c * form` overloads are not provided. For terms with the same domain
and test output, splitting or combining integrands gives the same operator,
up to floating-point roundoff. For example:

```cpp
auto combined = integrate(cells, dot(grad(u), grad(v)) + u * v);
auto split = integrate(cells, dot(grad(u), grad(v)))
           + integrate(cells, u * v);
```

Separate integrands may target different test names, but a single combined
integrand may not. Separate integrals also remain separate evaluation and
contraction contributions, so equivalent forms need not have identical cost.

For $R(x)=A(x)+c$, `lhs(R)` produces $A$ and `rhs(R)` produces $-c$.
Thus a residual written with `-f * v`, as in the first example, yields the
positive load $\int f v$. A pure linear form has a zero `rhs` that still
retains its output space. Applying `rhs` to an input-free load negates it;
`rhs` is not an idempotent accessor.

Extraction requires provable joint affinity in all active trials. Products
such as `u * p` are nonlinear when both symbols are active trials. Opaque
callbacks reading active fields and active-field-dependent upwind selectors
also prevent affine extraction. Runtime zeros or numerical cancellations
do not establish affinity.

A nonlinear residual can instead be evaluated directly:

```cpp
auto nonlinear = integrate(cells, dot(grad(u), grad(v)) + u * u * u * v);
auto residual_operator = MakeGenericOperator<SerialKernelConfiguration>(
   nonlinear, context, rules);
residual_operator(x, y);
```

See [form sums](sumformexpr.hpp) and [affine extraction](affinesplit.hpp).

## Evaluation and assembly

The numerical entry points take a form, its context, and an integration rule.
The kernel policy selects execution; the rule selects quadrature. Choose
quadrature appropriate to the polynomial degrees, geometry, and coefficients
of your problem.

| API | Behavior |
| --- | --- |
| `MakeGenericOperator<Policy>(form, context, rules)` | Construct a matrix-free residual operator. |
| `op.InputSize()`, `op.OutputSize()` | Required algebraic vector extents. |
| `op(x, y)` | Evaluate a form with active inputs, replacing the output. |
| `op(y)` | Evaluate an input-free form such as a load, replacing the output. |
| `GenericAssembly<Format, Policy>(form, context, rules)` | Assemble a supported linear form into a sparse matrix. |

Local facet quadrature must be compatible with the mesh's face permutations
and reflections. Matching Gauss or Gauss-Lobatto rules on exchanged axes
satisfy this condition. Optional `ValidateOperator` checks can verify it;
they do not run automatically. See [quadrature validation](../MatrixFreeOperators/GenericOperator/README.md#optional-quadrature-validation)
for the policy and alternatives when a rule is incompatible.

For example, the first example's linear operator can also be assembled:

```cpp
auto matrix = GenericAssembly<MatrixAssemblyType::CSR, SerialKernelConfiguration>(
   lhs(residual), context, rules);
```

Sparse assembly currently requires one named trial/test pair and does not
support callbacks reading active trial fields. Extract `lhs` to separate
loads before assembly. Available formats include BSR, SGBSR, RawCOO, COO,
CSR, CSC, and optional HypreCSR; restrictions vary by format and domain.
See [sparse matrix assembly](../MatrixAssembly/README.md) for those details
and [operator execution](../MatrixFreeOperators/GenericOperator/README.md)
for additive evaluation, validation, and custom execution contexts.

The same DSL is used with supported CPU and GPU kernel policies. Device
execution requires a CUDA or HIP build, suitable device-accessible data, and
device-callable coefficient functions. A serial example does not select a
GPU policy automatically.

## Current scope

The tables distinguish executable operations from syntax that lacks execution
support, and note restrictions on supported expressions. In particular:

- Scalar and componentwise vector finite-element mappings are supported.
  `div` and `curl` do not add geometric Piola, H(div), or H(curl) mappings.
- Derivatives of arbitrary expressions and arbitrary nonlinear dependence
  on test fields are not supported.
- `InterfaceFacets<"left", "right">` is declared but has no generic execution
  support. Use the supported domains listed above.
- Test-free scalar integrals are not reductions through the residual
  operator API; there is no implemented `ComputeIntegral` entry point.

Examples of supported behavior are also covered by the
[weak-form unit tests](../../../../tests/unit) and
[integration tests](../../../../tests/integration/gendil), including
[anisotropic diffusion](../../../../tests/integration/gendil/test-anisotropic-diffusion-coefficient.cpp),
[boundary upwinding](../../../../tests/unit/test-boundary-upwind.cpp),
[gradient upwinding](../../../../tests/integration/gendil/test-upwind-gradients.cpp),
[matrix products and pullbacks](../../../../tests/integration/gendil/test-matrix-products.cpp), and
[facet geometry and split/combined forms](../../../../tests/integration/gendil/test-facet-size-inputs.cpp).
