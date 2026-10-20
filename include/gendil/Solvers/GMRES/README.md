# Solving linear systems with GMRES

GMRES approximates the solution of a fixed linear system `A(solution) = rhs`.
It builds a Krylov basis, finds a correction in that small space, and restarts
when the basis reaches the chosen length. The same algorithm supports an
ordinary solve, fixed right preconditioning, and flexible right preconditioning.

## Start with one solve

Suppose `A` is a square GenDiL operator or sparse matrix and `rhs` is a supplied
`Vector` of the matching length. This starts from a zero initial guess:

```cpp
#include <gendil/Solvers/GMRES/gmres.hpp>
using namespace gendil;

GMRESOptions options;
options.restart = 20;
options.relative_tolerance = 1.e-10;

Vector solution(GetVectorSize(rhs));
solution = 0.0;
auto result = GMRES(A, rhs, options, solution);
```

`solution` receives the approximation. To start from another guess, fill it
before the call instead of assigning zero. The result reports:

- `success`: whether GMRES accepted a recomputed true residual `rhs - A(solution)`
  (an empty host system succeeds immediately).
- `iterations`: attempted Arnoldi steps, including failed attempts, across all
  restart cycles. Residual and correction applications are not counted.
- `relative_error`: the residual norm divided by the RHS norm, or the absolute
  residual norm when the RHS is zero. Invalid norms or an unrepresentable ratio
  produce infinity.

The stopping bound is `max(absolute_tolerance, relative_tolerance * norm(rhs))`;
the implementation checks it without forming the potentially overflowing
product. A zero iteration budget still computes the initial residual, leaves
the solution unchanged, and may report nonconvergence with zero iterations.
Modern entry points require a positive restart length.
An empty host system returns success with zero iterations and error, without
invoking numerical callbacks.

This one-shot overload owns its temporary storage and selects Euclidean
arithmetic with a scaled norm, batched projection, and fused basis combination.
Modified Gram–Schmidt is the default; pass `ClassicalGramSchmidtTwoPass{}` as
the last argument to select two-pass classical Gram–Schmidt.

## Reuse storage and choose an inner product

For repeated host-controlled solves, keep a workspace and a dot operation alive.
Using the same `A`, `rhs`, `solution`, and `options` as above:

```cpp
GMRESWorkspace workspace;
EuclideanDot<HostKernelConfiguration> dot{workspace.reduction_workspace};

solution = 0.0;
auto result = GMRES(A, rhs, solution, dot, options, workspace);
```

The owner prepares algorithm storage as needed. At unchanged vector length,
sufficient capacities reuse allocations; changing the vector length can
reconstruct storage. The owner creates fresh buffer descriptors after
preparation. Existing borrowed descriptors do not automatically rebind after
storage changes. See the [workspace header](Workspace/workspace.hpp).

The explicit-dot overload borrows the exact dot object, including mutable or
noncopyable callables. It derives its norm through `NormFromDot`, the checked
`sqrt(dot(x,x))`, and projects through that same dot. It does not discover a
custom norm method. In particular, a self-dot can overflow even when the norm
is representable; the one-shot overload's scaled norm avoids that intermediate
overflow.

The example's CPU dot needs no reduction entries. Host-controlled device
arithmetic needs caller-provisioned scratch: the `Required*WorkspaceSize`
helpers in [vector reductions](../../Algebra/Vector/reductions.hpp) describe
the supplied operations' requirements. Algorithm preparation does not resize
the optional `workspace.reduction_workspace` or custom callback resources.

### Right preconditioning

Let `P(input, output)` be a supplied approximate inverse action that replaces
`output`. With the dot and workspace above:

```cpp
solution = 0.0;
auto fixed = GMRES(A, P, rhs, solution, dot, options, workspace);

solution = 0.0;
auto flexible = FGMRES(A, P, rhs, solution, dot, options, workspace,
                       ClassicalGramSchmidtTwoPass{});
```

Fixed-right GMRES requires the same linear `P` throughout the solve. It applies
`P` during each expansion and once more to the combined correction. Flexible
GMRES stores each produced direction and combines those directions directly;
it never reapplies `P` to a combined correction. A changing binary preconditioner
keeps its own state or application counter. The system `A` remains fixed and
linear in both variants.

### Supply norm, projection, and combination explicitly

The advanced unpreconditioned host overload accepts these three operations
separately. For the CPU Euclidean dot already defined:

```cpp
auto norm = [&](const auto& x) { return ScaledEuclideanNorm(dot, x); };
auto project = [&](auto& basis, Integer count, const auto& x, auto& coefficients)
{
   ProjectEuclidean(dot, basis, count, x, coefficients);
};
auto combine = [](auto& basis, Integer count, const auto& coefficients,
                  Real alpha, auto& output)
{
   HostDrivenFusedCombine<false>(basis, count, coefficients, alpha, output);
};
solution = 0.0;
auto result = GMRES(A, rhs, solution, dot, options, workspace,
                    norm, project, combine, ClassicalGramSchmidtTwoPass{});
```

Norm and projection must be consistent with the supplied inner product.
Projection writes `dot(basis(j), x)` without changing `x`; combination adds
`alpha * sum_j coefficients[j] * basis(j)` to the output and is independent of
the metric. Callables need only accept the actual operands used by the solve.
Device fused combination has an explicit coefficient-scratch overload.

## Borrow buffers for a local solve

Use `TieGMRESBuffers(basis, work, directions, H, cosines, sines, g, y)` when
storage already exists or the solve runs inside a kernel. The basis holds
Krylov vectors; `work` holds an expanded vector; `H`, the rotations, `g`, and
`y` hold the small least-squares problem. The binder references all eight
named operands, including their descriptor objects. It does not copy entries,
select representations, or validate capacity.

This independent **CPU thread-local example** solves the identity system.
It uses private arrays, not cooperative shared storage:

```cpp
#include <gendil/Solvers/GMRES/gmres.hpp>
#include <array>
using namespace gendil;

constexpr size_t vector_size = 2;
constexpr Integer restart = 2;
constexpr size_t leading_dimension = restart + 1;
constexpr size_t dense_size = leading_dimension * restart + 4 * restart + 1;

using Context = KernelContext<HostKernelConfiguration, 0>;
std::array<Real, Context::shared_memory_block_size> arena{};
Context context(arena.data());
std::array<Real, vector_size> rhs{2, -1}, solution{}, work{};
std::array<std::array<Real, vector_size>, restart + 1> basis_storage{};
std::array<Real, dense_size> dense_storage{};
auto basis = MakeVectorView(basis_storage.data(), basis_storage.size());
auto scalars = MakeVectorView(dense_storage.data(), dense_storage.size());
auto dense = MakeGMRESPackedDense(scalars, restart, leading_dimension);
NoDirections directions;
auto buffers = TieGMRESBuffers(basis, work, directions, dense.hessenberg,
   dense.cosines, dense.sines, dense.reduced_rhs, dense.coefficients);

auto identity = [](const auto& input, auto& output) { output = input; };
auto dot = [&](const auto& x, const auto& y) { return LocalDot(context, x, y); };
GMRESOptions options;
options.restart = restart;
auto result = GMRES(context, identity, rhs, solution, dot, options, buffers);
// On success, solution approximates {2, -1}.
```

Keep every bound descriptor and its storage alive and stable during the call.
Each operand needs only the active restart capacity; capacities need not all
match. `NoDirections` is not inspected by an unpreconditioned solve. Fixed-right
and flexible solves need one and `restart` direction vectors respectively.

The [buffer header](Workspace/workspaceview.hpp) specifies the operand lengths
and packed layout. Packed offsets and the leading dimension count logical
entries of the supplied scalar view, preserving its origin and scalar stride.
A smaller active restart does not change the packed reservation or offsets.

### Execution and completion contracts

Host-controlled solves can launch CPU or GPU vector operations. Native-context
calls execute locally instead: thread-local solves use private storage, while
cooperative solves require shared dense scratch, uniform participation, and
`batch_size == 1`. The example above is not a cooperative GPU recipe.

Each callback completes the dependencies needed by the next operation. Maps and
preconditioners must handle their internal cross-participant reads and writes;
the driver does not append a compensating barrier after their application.
Scalar reductions return the same value to every participant. Local return
completes solver-buffer reads on success and failure, permitting scratch reuse.
Host device operations may complete through stream ordering rather than a
host-blocking synchronization.

Modern host entry points check declared logical dimensions before numerical
callbacks. Local CPU-debug entry points assert them; local release/device
lengths, residency, shared-arena capacity, and buffer placement are caller
preconditions. No validator discovers hidden allocation lengths behind raw
pointers. Operands must not overlap where the algorithm needs distinct values,
and simultaneous solves must not share mutable scratch.

The [reduction scratch header](Workspace/reductionworkspace.hpp) distinguishes
optional host-managed arithmetic storage from temporary cooperative shared
arena space. These are separate from the persistent basis and dense operands;
custom callbacks remain responsible for their own scratch requirements.

## Reading the implementation

Start at the [public interface](gmres.hpp), then follow the
[restart algorithm](restartedgmres.hpp), [Arnoldi expansion](arnoldi.hpp), and
[reduced least squares](leastsquares.hpp). Orthogonalization leaves an
unnormalized remainder; the restart controller decides dependence and
normalization. Reduced residual estimates can end a cycle but never establish
final convergence. Failed operations can leave scratch partially updated.

[Drivers/drivers.hpp](Drivers/drivers.hpp) collects the concrete compositions:
host preparation/application, local execution, and legacy adaptation. Admission
predicates and column-view utilities are separate from those compositions.
General vector arithmetic remains in Algebra, without solver dependencies.

[Shared solver utilities](../iterativesolverutilities.hpp) provide norm checks,
relative/absolute convergence tests, result formatting, and true-residual
formation for both CG and GMRES. The drivers select the norm after forming the
residual, so sharing adds no reductions or synchronization. Each solver retains
its own tolerance translation, iteration counting, and workspace handling.

The legacy `GMRES_no_alloc` entry point uses the same restarted algorithm with
split arrays and its historical arithmetic conventions. Non-View assignment
must copy entries into existing storage, without rebinding, resizing, or
allocating. Known capacities are checked; unknown ones remain preconditions.
Zero budget or zero restart still applies the system and computes the initial
residual using work storage and one basis vector, but needs no dense arrays and
does not change the solution. Unlike modern options, this adapter accepts zero
restart. For a zero RHS, its single tolerance is an absolute bound.
