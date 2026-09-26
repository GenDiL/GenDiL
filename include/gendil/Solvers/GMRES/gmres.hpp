// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Solve a linear system using restarted GMRES or flexible GMRES.
 *
 * Given a square operator or matrix A and a GenDiL Vector rhs, start from zero:
 * @code
 * GMRESOptions options;
 * options.restart = 20;
 * Vector solution(GetVectorSize(rhs));
 * solution = 0.0;
 * auto result = GMRES(A, rhs, options, solution);
 * @endcode
 * The existing solution values are the initial guess; the same vector receives
 * the approximation. result.success reports convergence, result.iterations
 * counts attempted Arnoldi steps (including failed attempts), and
 * result.relative_error reports the final residual norm divided by the RHS
 * norm, or the absolute residual norm for a zero RHS. Invalid norms or an
 * unrepresentable quotient produce infinite reported error. A zero iteration
 * budget still checks the initial residual and does not change the solution.
 * An empty host system succeeds immediately with zero iterations and error,
 * without invoking numerical callbacks.
 *
 * This one-shot overload owns its scratch and selects Euclidean arithmetic
 * with a scaled norm. For repeated solves, supply a workspace and dot product:
 * GMRES(A, rhs, solution, dot, options, workspace). Explicit-dot overloads use
 * NormFromDot, the checked sqrt(dot(x,x)), rather than discovering a norm from
 * the dot object. The advanced host overload accepts a norm, projection, and
 * combination explicitly. Their contracts are described beside that overload.
 *
 * Fixed right preconditioning uses GMRES(A,P,rhs,solution,dot,options,workspace);
 * FGMRES uses that call shape for a changing binary P(input,output). Both keep
 * the system fixed and linear. ModifiedGramSchmidt is the default; supply
 * ClassicalGramSchmidtTwoPass as the final tag to select two-pass classical
 * orthogonalization. Reduced coefficients use Real arithmetic.
 *
 * Native-context calls prepend KernelContext and borrow already-formed buffers.
 * Maps and preconditioners must complete their own outputs. Cooperative solves
 * require shared dense storage, uniform participation and batch_size == 1.
 * Local release/device shapes and arena capacity are caller preconditions.
 *
 * For nonempty systems, success requires a recomputed true residual; reduced
 * estimates only decide when to end a cycle. Residual/correction applications
 * do not count as Arnoldi steps. Calls borrow the exact supplied callback objects,
 * including mutable and noncopyable ones. Keep callables, descriptors, and
 * storage alive throughout the synchronous solve. Inputs and solution must not
 * overlap solver scratch.
 */
#include "gendil/Algebra/Vector/localoperations.hpp"
#include "gendil/Algebra/Vector/reductions.hpp"
#include "gendil/Solvers/GMRES/Drivers/drivers.hpp"
#include "gendil/Solvers/GMRES/Workspace/workspace.hpp"
namespace gendil
{
/** @brief Reuse host-controlled scratch with the caller's inner product.
 *
 * dot(x,y) defines the orthogonalization metric and NormFromDot supplies its
 * norm. The map must satisfy VectorMap for the actual input/work operand types.
 * Workspace may be an owner, prepared as needed, or already-bound buffers whose
 * declared dimensions are checked before numerical callbacks. Kernel policy
 * selects the memory space of vector updates, not the dot's implementation.
 * The caller supplies compatible arithmetic and any reduction scratch it needs.
 */
template <class Policy = gendil::HostKernelConfiguration, class A, class Rhs, class X,
          class Dot, class Workspace, class OrthogonalizationMethod = ModifiedGramSchmidt>
   requires gmres_detail::IsGMRESOrthogonalization<OrthogonalizationMethod> &&
            gmres_detail::SupportsHostSolve<
                Policy, gmres_detail::PreconditioningMode::None, A, Rhs, X, Workspace>
auto GMRES(const A &system, const Rhs &rhs, X &solution, Dot &&dot,
           const GMRESOptions &options, Workspace &workspace,
           OrthogonalizationMethod orth = {})
{
   auto norm = [&](const auto &v)
   {
      return NormFromDot(dot, v);
   };
   auto project =
       [&dot](auto &basis, Integer count, const auto &input, auto &coefficients)
   {
      ProjectFromDot(dot, basis, count, input, coefficients);
   };
   auto combine = [](auto &basis, Integer count, const auto &c, Real alpha, auto &out)
   {
      Combine<gendil::is_device_configuration_v<Policy>>(basis, count, c, alpha, out);
   };
   gmres_detail::NoPreconditioner preconditioner;
   return gmres_detail::HostSolve<Policy, gmres_detail::PreconditioningMode::None>(
       system, preconditioner, rhs, solution, dot, options, workspace, norm, project,
       combine, orth);
}

/** @brief Solve with a fixed linear right preconditioner and an explicit dot.
 *
 * P(input,output) applies an approximate inverse: expansion computes A*P(v),
 * and each completed correction applies P once more to the combined basis
 * vector. P must therefore remain the same linear operation throughout a solve.
 * Workspace, metric, and lifetime rules are those of the explicit-dot GMRES.
 */
template <class Policy = gendil::HostKernelConfiguration, class A, class P, class Rhs,
          class X, class Dot, class Workspace,
          class OrthogonalizationMethod = ModifiedGramSchmidt>
   requires gmres_detail::IsGMRESOrthogonalization<OrthogonalizationMethod> &&
            gmres_detail::SupportsHostSolve<Policy,
                                            gmres_detail::PreconditioningMode::FixedRight,
                                            A, Rhs, X, Workspace>
auto GMRES(const A &system, P &&preconditioner, const Rhs &rhs, X &solution, Dot &&dot,
           const GMRESOptions &options, Workspace &workspace,
           OrthogonalizationMethod orth = {})
{
   auto norm = [&](const auto &v)
   {
      return NormFromDot(dot, v);
   };
   auto project =
       [&dot](auto &basis, Integer count, const auto &input, auto &coefficients)
   {
      ProjectFromDot(dot, basis, count, input, coefficients);
   };
   auto combine = [](auto &basis, Integer count, const auto &c, Real alpha, auto &out)
   {
      Combine<gendil::is_device_configuration_v<Policy>>(basis, count, c, alpha, out);
   };
   return gmres_detail::HostSolve<Policy, gmres_detail::PreconditioningMode::FixedRight>(
       system, preconditioner, rhs, solution, dot, options, workspace, norm, project,
       combine, orth);
}

/** @brief Solve while allowing the right preconditioner to change between calls.
 *
 * Store each P(v) direction and combine those stored vectors in the correction;
 * P is never reapplied to the combined correction. The binary callable owns any
 * iteration-dependent state. The system remains fixed and linear, and workspace
 * and metric rules match explicit-dot GMRES.
 */
template <class Policy = gendil::HostKernelConfiguration, class A, class P, class Rhs,
          class X, class Dot, class Workspace,
          class OrthogonalizationMethod = ModifiedGramSchmidt>
   requires gmres_detail::IsGMRESOrthogonalization<OrthogonalizationMethod> &&
            gmres_detail::SupportsHostSolve<
                Policy, gmres_detail::PreconditioningMode::FlexibleRight, A, Rhs, X,
                Workspace>
auto FGMRES(const A &system, P &&preconditioner, const Rhs &rhs, X &solution, Dot &&dot,
            const GMRESOptions &options, Workspace &workspace,
            OrthogonalizationMethod orth = {})
{
   auto norm = [&](const auto &v)
   {
      return NormFromDot(dot, v);
   };
   auto project =
       [&dot](auto &basis, Integer count, const auto &input, auto &coefficients)
   {
      ProjectFromDot(dot, basis, count, input, coefficients);
   };
   auto combine = [](auto &basis, Integer count, const auto &c, Real alpha, auto &out)
   {
      Combine<gendil::is_device_configuration_v<Policy>>(basis, count, c, alpha, out);
   };
   return gmres_detail::HostSolve<Policy,
                                  gmres_detail::PreconditioningMode::FlexibleRight>(
       system, preconditioner, rhs, solution, dot, options, workspace, norm, project,
       combine, orth);
}

/** @brief Choose all metric and basis-combination operations for a host solve.
 *
 * norm(x) returns a norm consistent with dot. project(basis,count,x,c) writes
 * c[j] = dot(basis(j),x) for the unchanged x. combine(basis,count,c,alpha,out)
 * adds alpha * sum_j c[j]*basis(j) to out, independently of the metric.
 * Each operation completes the outputs needed by the next solver operation;
 * host-controlled device calls may use stream ordering rather than host blocking.
 * Callables need only support the actual operand types of this solve, not every
 * vector representation. All callbacks are borrowed, not copied or discovered.
 */
template <class Policy = gendil::HostKernelConfiguration, class A, class Rhs, class X,
          class Dot, class Norm, class Project, class Combine, class Workspace,
          class OrthogonalizationMethod>
   requires gmres_detail::IsGMRESOrthogonalization<OrthogonalizationMethod> &&
            gmres_detail::SupportsHostSolve<
                Policy, gmres_detail::PreconditioningMode::None, A, Rhs, X, Workspace>
auto GMRES(const A &system, const Rhs &rhs, X &solution, Dot &&dot,
           const GMRESOptions &options, Workspace &workspace, Norm &&norm,
           Project &&project, Combine &&combine, OrthogonalizationMethod orth)
{
   gmres_detail::NoPreconditioner preconditioner;
   return gmres_detail::HostSolve<Policy, gmres_detail::PreconditioningMode::None>(
       system, preconditioner, rhs, solution, dot, options, workspace, norm, project,
       combine, orth);
}

/** @brief Solve inside a native execution context using caller-owned buffers.
 *
 * Use private storage for a thread-local solve or shared dense arrays and
 * uniform participation for a cooperative solve. No owning workspace is prepared
 * here. dot must be callable in this execution context and return the same scalar
 * to every participant. Structural assertions run only on CPU debug paths;
 * release/device sizes, residency, and arena capacity are caller preconditions.
 * Maps complete their own internal dependencies; return completes cooperative
 * solver-buffer reads so scratch can be reused immediately.
 */
template <class Context, class A, class Rhs, class X, class Dot, class B,
          class OrthogonalizationMethod = ModifiedGramSchmidt>
   requires gmres_detail::IsGMRESOrthogonalization<OrthogonalizationMethod> &&
            requires { typename Context::kernel_configuration_type; } &&
            gmres_detail::SupportsLocalSolve<gmres_detail::PreconditioningMode::None, A,
                                             X, B>
GENDIL_HOST_DEVICE auto GMRES(Context &context, const A &system, const Rhs &rhs,
                              X &solution, Dot &&dot, const GMRESOptions &options,
                              B &buffers, OrthogonalizationMethod orth = {})
{
   auto norm = [&] GENDIL_HOST_DEVICE(const auto &v)
   {
      return NormFromDot(dot, v);
   };
   auto project = [&context, &dot] GENDIL_HOST_DEVICE(
                      auto &basis, Integer count, const auto &input, auto &coefficients)
   {
      LocalProject(context, dot, basis, count, input, coefficients);
   };
   auto combine = [&] GENDIL_HOST_DEVICE(auto &basis, Integer count, const auto &c,
                                         Real alpha, auto &out)
   {
      LocalCombine(context, basis, count, c, alpha, out);
   };
   gmres_detail::NoPreconditioner preconditioner;
   return gmres_detail::LocalSolve<gmres_detail::PreconditioningMode::None>(
       context, system, preconditioner, rhs, solution, dot, options, buffers, norm,
       project, combine, orth);
}

/** @brief Apply fixed-right GMRES with the native-context completion contract.
 * Requires one direction vector; P must satisfy the fixed linear-map contract
 * and complete its outputs for the next collective operation.
 */
template <class Context, class A, class P, class Rhs, class X, class Dot, class B,
          class OrthogonalizationMethod = ModifiedGramSchmidt>
   requires gmres_detail::IsGMRESOrthogonalization<OrthogonalizationMethod> &&
            requires { typename Context::kernel_configuration_type; } &&
            gmres_detail::SupportsLocalSolve<
                gmres_detail::PreconditioningMode::FixedRight, A, X, B>
GENDIL_HOST_DEVICE auto GMRES(Context &context, const A &system, P &&preconditioner,
                              const Rhs &rhs, X &solution, Dot &&dot,
                              const GMRESOptions &options, B &buffers,
                              OrthogonalizationMethod orth = {})
{
   auto norm = [&] GENDIL_HOST_DEVICE(const auto &v)
   {
      return NormFromDot(dot, v);
   };
   auto project = [&context, &dot] GENDIL_HOST_DEVICE(
                      auto &basis, Integer count, const auto &input, auto &coefficients)
   {
      LocalProject(context, dot, basis, count, input, coefficients);
   };
   auto combine = [&] GENDIL_HOST_DEVICE(auto &basis, Integer count, const auto &c,
                                         Real alpha, auto &out)
   {
      LocalCombine(context, basis, count, c, alpha, out);
   };
   return gmres_detail::LocalSolve<gmres_detail::PreconditioningMode::FixedRight>(
       context, system, preconditioner, rhs, solution, dot, options, buffers, norm,
       project, combine, orth);
}

/** @brief Apply flexible-right GMRES with the native-context completion contract.
 * Requires one stored direction per active restart step. The binary P may change
 * its action between calls but must complete each output before it is consumed.
 */
template <class Context, class A, class P, class Rhs, class X, class Dot, class B,
          class OrthogonalizationMethod = ModifiedGramSchmidt>
   requires gmres_detail::IsGMRESOrthogonalization<OrthogonalizationMethod> &&
            requires { typename Context::kernel_configuration_type; } &&
            gmres_detail::SupportsLocalSolve<
                gmres_detail::PreconditioningMode::FlexibleRight, A, X, B>
GENDIL_HOST_DEVICE auto FGMRES(Context &context, const A &system, P &&preconditioner,
                               const Rhs &rhs, X &solution, Dot &&dot,
                               const GMRESOptions &options, B &buffers,
                               OrthogonalizationMethod orth = {})
{
   auto norm = [&] GENDIL_HOST_DEVICE(const auto &v)
   {
      return NormFromDot(dot, v);
   };
   auto project = [&context, &dot] GENDIL_HOST_DEVICE(
                      auto &basis, Integer count, const auto &input, auto &coefficients)
   {
      LocalProject(context, dot, basis, count, input, coefficients);
   };
   auto combine = [&] GENDIL_HOST_DEVICE(auto &basis, Integer count, const auto &c,
                                         Real alpha, auto &out)
   {
      LocalCombine(context, basis, count, c, alpha, out);
   };
   return gmres_detail::LocalSolve<gmres_detail::PreconditioningMode::FlexibleRight>(
       context, system, preconditioner, rhs, solution, dot, options, buffers, norm,
       project, combine, orth);
}

/** Solve with legacy split buffers and a callable-only map.
 *
 * dot supplies NormFromDot; map(input,output) replaces output and may accept
 * mutable input or return a value that is ignored. Other legacy arithmetic uses
 * entry-copy assignment, division, and the available three-argument ADL AXPY.
 * The caller supplies all storage. Known logical lengths are checked before
 * callbacks; unsized storage must satisfy the capacities documented by
 * gmres_detail::LegacySolve.
 * Non-View assignment must copy entries without rebinding, resizing, or allocating.
 * For zero rhs, tolerance is absolute. Zero restart/budget still needs a work
 * vector and one basis vector to compute the initial residual; dense arrays may
 * be empty. The solution remains unchanged and the iteration count is zero.
 */
template <class Map, class RHS, class Dot, class VectorArray, class VectorType,
          class Array, class Orthogonalization = ModifiedGramSchmidt>
auto GMRES_no_alloc(const Map &map, const RHS &rhs, Dot &&dot, Integer max_iters,
                    Integer restart, Real tolerance, VectorType &solution,
                    VectorArray &basis, VectorType &work, Array &hessenberg,
                    Array &cosine, Array &sine, Array &reduced_rhs,
                    Array &reduced_solution, Orthogonalization orthogonalization = {})
{
   return gmres_detail::LegacySolve(map, rhs, dot, max_iters, restart, tolerance,
                                    solution, basis, work, hessenberg, cosine, sine,
                                    reduced_rhs, reduced_solution, orthogonalization);
}

/** @brief Solve once with internally owned scratch and Euclidean arithmetic.
 *
 * Uses a scaled Euclidean norm, batched projection, and fused basis combination.
 * KernelPolicy selects host-controlled CPU or device vector execution; it does
 * not move the restart loop inside a kernel. Algorithm and arithmetic scratch
 * are scoped to this call. Use the explicit-dot overloads to reuse resources.
 */
template <class Policy = HostKernelConfiguration, class Map, class RHS, class Solution,
          class Orthogonalization = ModifiedGramSchmidt>
   requires gmres_detail::IsGMRESOrthogonalization<Orthogonalization> &&
            gmres_detail::SupportsHostSolve<Policy,
                                            gmres_detail::PreconditioningMode::None, Map,
                                            RHS, Solution, GMRESWorkspace>
auto GMRES(const Map &map, const RHS &rhs, const GMRESOptions &options,
           Solution &solution, Orthogonalization orthogonalization = {})
{
   GMRESWorkspace workspace;
   const size_t count = std::min(options.restart, GetVectorSize(solution));
   const size_t size = GetVectorSize(solution);
   constexpr bool device = is_device_configuration_v<Policy>;
   Vector scratch(device ? std::max(RequiredNormReductionWorkspaceSize(size),
                                    RequiredProjectionReductionWorkspaceSize(size, count))
                         : 0);
   EuclideanDot<Policy> dot{scratch};
   auto norm = [&](const auto &input)
   {
      return ScaledEuclideanNorm(dot, input);
   };
   auto project = [&](auto &basis, Integer n, const auto &input, auto &c)
   {
      ProjectEuclidean(dot, basis, n, input, c);
   };
   auto combine = [&](auto &basis, Integer n, const auto &c, Real alpha, auto &output)
   {
      if constexpr (device)
      {
         HostDrivenFusedCombine<true>(basis, n, c, alpha, output, scratch);
      }
      else
      {
         HostDrivenFusedCombine<false>(basis, n, c, alpha, output);
      }
   };
   return GMRES<Policy>(map, rhs, solution, dot, options, workspace, norm, project,
                        combine, orthogonalization);
}

} // namespace gendil
