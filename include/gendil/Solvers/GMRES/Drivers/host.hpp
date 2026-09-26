// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Run GMRES from the host, with vector operations on the CPU or GPU.
 *
 * Bind system applications, vector arithmetic, and reduced solves to the shared
 * restart loop. HostSolve checks dimensions and prepares owning scratch before
 * entering HostSolvePrepared; borrowed scratch retains its supplied layout.
 */

#include "gendil/Algebra/Vector/localoperations.hpp"
#include "gendil/Algebra/vectormapconcept.hpp"
#include "gendil/Algebra/Vector/operations.hpp"
#include "gendil/Solvers/GMRES/Drivers/constraints.hpp"
#include "gendil/Solvers/GMRES/Drivers/operands.hpp"
#include "gendil/Solvers/GMRES/Workspace/workspace.hpp"
#include "gendil/Solvers/GMRES/arnoldi.hpp"
#include "gendil/Solvers/GMRES/leastsquares.hpp"
#include "gendil/Solvers/GMRES/restartedgmres.hpp"
#include "gendil/Utilities/KernelContext/kernelconfiguration.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"
#include <algorithm>
#include <type_traits>

namespace gendil::gmres_detail
{

/** Update the solution using already-prepared operands and supplied operations.
 * Options, system dimensions, and active buffer extents must already be valid;
 * this internal path performs no preparation or metadata checks. Vector storage
 * must support the selected policy, while reduced-system entries are host-accessible.
 * Callbacks are borrowed, must complete outputs before consumption, and must use
 * a consistent norm/projection geometry. Their bodies are not validated here.
 */
template <class Policy, PreconditioningMode mode, class A, class P, class Rhs, class X,
          class Dot, class B, class Norm, class Project, class Combine,
          class OrthogonalizationMethod>
auto HostSolvePrepared(const A &system, P &preconditioner, const Rhs &rhs, X &solution,
                       Dot &dot, const GMRESOptions &active_options, B &buffers,
                       Norm &norm, Project &project, Combine &combine,
                       OrthogonalizationMethod)
{
   constexpr bool device = is_device_configuration_v<Policy>;
   static_assert(IsGMRESOrthogonalization<OrthogonalizationMethod>);
   // Vector operations.
   auto apply_system = [&](const auto &input, auto &output)
   {
      using gendil::Apply;
      Apply(system, input, output);
   };
   auto copy_vector = [](const auto &input, auto &output)
   {
      gendil::CopyVector<device>(input, output);
   };
   auto axpy = [](Real alpha, const auto &input, auto &output)
   {
      gendil::Axpy<device>(alpha, input, output);
   };
   auto divide_vector = [](auto &output, Real length)
   {
      Divide<device>(output, length);
   };
   // Reduced-system operations.
   auto store_coefficient = [](auto &c, Integer i, Real value)
   {
      StoreCoefficient(c, i, value);
   };
   auto accumulate_coefficients = [](auto &destination, const auto &source, Integer count)
   {
      auto x = gendil::MakeView(gendil::ReferenceContainer{source},
                                gendil::VectorLayout{count});
      auto y = gendil::MakeView(gendil::ReferenceContainer{destination},
                                gendil::VectorLayout{count});
      gendil::HostKernelConfiguration host;
      gendil::Axpy(host, Real(1), x, y);
   };
   auto initialize_reduced_problem = [&](Real beta)
   {
      InitializeReducedLeastSquares(buffers.cosines, buffers.sines, buffers.reduced_rhs,
                                    buffers.coefficients, active_options.restart, beta);
   };
   auto update_reduced_qr = [&](Integer j, Real next)
   {
      return UpdateReducedQR(buffers.hessenberg, buffers.cosines, buffers.sines,
                             buffers.reduced_rhs, j, next);
   };
   auto solve_reduced_system = [&](Integer count)
   {
      return BackSubstitute(buffers.hessenberg, buffers.reduced_rhs, buffers.coefficients,
                            count);
   };

   // Expansion, correction, and orthogonalization.
   auto expand_search_direction = [&](Integer column)
   {
      auto &&basis_vector = buffers.basis(column);
      if constexpr (mode == PreconditioningMode::None)
      {
         apply_system(basis_vector, buffers.work);
      }
      else
      {
         const Integer direction_index =
             mode == PreconditioningMode::FixedRight ? 0 : column;
         auto &&direction = buffers.directions(direction_index);
         preconditioner(basis_vector, direction);
         apply_system(direction, buffers.work);
      }
   };
   auto apply_correction = [&](Integer count)
   {
      if constexpr (mode == PreconditioningMode::None)
      {
         combine(buffers.basis, count, buffers.coefficients, Real(1), solution);
      }
      else if constexpr (mode == PreconditioningMode::FixedRight)
      {
         gendil::Zero<device>(buffers.work);
         combine(buffers.basis, count, buffers.coefficients, Real(1), buffers.work);
         auto &&direction = buffers.directions(0);
         preconditioner(buffers.work, direction);
         axpy(Real(1), direction, solution);
      }
      else
      {
         combine(buffers.directions, count, buffers.coefficients, Real(1), solution);
      }
   };
   auto orthogonalize = [&](Integer column)
   {
      auto hessenberg_column = MakeHessenbergColumnView(buffers.hessenberg, column);
      if constexpr (std::is_same_v<OrthogonalizationMethod, ModifiedGramSchmidt>)
         return OrthogonalizeModifiedGramSchmidt(
             dot, axpy, store_coefficient, buffers.basis, column + 1, buffers.work,
             hessenberg_column);
      else
         return OrthogonalizeClassicalGramSchmidtTwoPass(
             project, combine, accumulate_coefficients, buffers.basis, column + 1,
             buffers.work, hessenberg_column, buffers.coefficients);
   };
   // Bind the phases consumed by the generic restart loop.
   auto &&residual_vector = buffers.basis(0);
   auto compute_true_residual_norm = [&]
   {
      return ComputeTrueResidualNorm(apply_system, rhs, solution, residual_vector,
                                     buffers.work, copy_vector, axpy, norm);
   };
   auto build_arnoldi_column = [&](Integer j)
   {
      return BuildArnoldiColumn(buffers.work, j, expand_search_direction, orthogonalize,
                                norm);
   };
   return RestartedGMRES(
       active_options,
       buffers,
       norm(rhs),
       compute_true_residual_norm,
       build_arnoldi_column,
       apply_correction,
       copy_vector,
       divide_vector,
       initialize_reduced_problem,
       update_reduced_qr,
       solve_reduced_system);
}

/** Validate host solve metadata, prepare scratch if owned, and run the solve.
 * Options and system/RHS/solution dimensions are checked before callbacks or
 * solution mutation. For nonempty systems the active restart is capped at n;
 * owning scratch is prepared, possibly allocating, then fresh views are formed.
 * Borrowed operands are checked independently without resizing or repacking.
 * Empty systems return success without numerical callbacks or scratch preparation.
 * Storage lifetimes, accessibility, aliasing, and callback semantics remain the
 * caller's responsibility; metadata validation does not establish them.
 */
template <class Policy, PreconditioningMode mode, class A, class P, class Rhs, class X,
          class Dot, class Workspace, class Norm, class Project, class Combine,
          class OrthogonalizationMethod>
auto HostSolve(const A &system, P &preconditioner, const Rhs &rhs, X &solution, Dot &dot,
               const GMRESOptions &options, Workspace &workspace, Norm &norm,
               Project &project, Combine &combine, OrthogonalizationMethod orth)
{
   const size_t n = GetVectorSize(solution);
   GENDIL_VERIFY(ValidOptions(options), "Invalid GMRES options.");
   GENDIL_VERIFY(GetVectorSize(rhs) == n && GetInputSize(system) == n &&
                     GetOutputSize(system) == n,
                 "Incompatible system dimensions.");
   if (n == 0)
   {
      return IterativeSolverResult{true, 0, 0};
   }
   auto active_options = options;
   active_options.restart = std::min(options.restart, n);
   if constexpr (std::is_same_v<Workspace, GMRESWorkspace>)
   {
      const size_t directions =
          mode == PreconditioningMode::None
              ? 0
              : (mode == PreconditioningMode::FixedRight ? 1 : active_options.restart);
      PrepareGMRESAlgorithmWorkspace(workspace, n, active_options.restart, directions);
      auto buffers = MakeGMRESWorkspaceView(workspace);
      GENDIL_VERIFY(ValidBuffers<mode>(buffers, n, active_options.restart),
                    "Insufficient GMRES buffers.");
      return HostSolvePrepared<Policy, mode>(system, preconditioner, rhs, solution, dot,
                                             active_options, buffers, norm, project,
                                             combine, orth);
   }
   else
   {
      GENDIL_VERIFY(ValidBuffers<mode>(workspace, n, active_options.restart),
                    "Insufficient GMRES buffers.");
      return HostSolvePrepared<Policy, mode>(system, preconditioner, rhs, solution, dot,
                                             active_options, workspace, norm, project,
                                             combine, orth);
   }
}

} // namespace gendil::gmres_detail
