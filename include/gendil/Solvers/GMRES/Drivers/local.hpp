// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Solve a local system within a thread or a cooperating kernel block.
 *
 * Use the native context and caller-provided operands directly: local arithmetic
 * and reduced solves feed the shared restart loop without host-side preparation.
 * A threaded context makes the solve collective across its physical block.
 */

#include "gendil/Algebra/Vector/localoperations.hpp"
#include "gendil/Algebra/vectormapconcept.hpp"
#include "gendil/Solvers/GMRES/Drivers/constraints.hpp"
#include "gendil/Solvers/GMRES/Drivers/operands.hpp"
#include "gendil/Solvers/GMRES/arnoldi.hpp"
#include "gendil/Solvers/GMRES/leastsquares.hpp"
#include "gendil/Solvers/GMRES/restartedgmres.hpp"
#include "gendil/Solvers/iterativesolverutilities.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include <type_traits>

namespace gendil::gmres_detail
{

/** Update a local solution using existing vector and reduced-system storage.
 * Debug CPU builds assert option and logical operand dimensions. Release and
 * device calls require those conditions on entry; this path neither prepares
 * storage nor caps the restart. Maps and callbacks must accept the actual
 * operands, and norms and projections must describe the same geometry.
 *
 * In a threaded context, every block participant must follow the same collective
 * sequence with uniform control decisions; batch_size must be one. Cooperative
 * dense scratch is shared, whereas thread-local dense scratch is private.
 * Maps and preconditioners own their internal completion before the next consumer:
 * the driver appends no barrier to an application. Supplied reductions and other
 * operations likewise complete their dependencies before subsequent use.
 * A terminal block synchronization completes solver-buffer reads before return
 * and reuse, including early failure; it does not repair missing earlier completion.
 */
template <PreconditioningMode mode, class Context, class A, class P, class Rhs, class X,
          class Dot, class B, class Norm, class Project, class Combine,
          class OrthogonalizationMethod>
GENDIL_HOST_DEVICE auto
LocalSolve(Context &context, const A &system, P &preconditioner, const Rhs &rhs,
           X &solution, Dot &dot, const GMRESOptions &options, B &buffers, Norm &norm,
           Project &project, Combine &combine, OrthogonalizationMethod)
{
   static_assert(
       !(gendil::is_threaded_v<Context> && Context::batch_size > 1),
       "Cooperative GMRES requires batch_size == 1: barriers span the physical block.");
   static_assert(IsGMRESOrthogonalization<OrthogonalizationMethod>);
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
   const size_t n = LogicalSize(solution);
   GENDIL_ASSERT(ValidOptions(options) && LogicalSize(rhs) == n &&
                     ValidBuffers<mode>(buffers, n, options.restart),
                 "Invalid local solve operands.");
#endif
   // Local callbacks inherit this function's host/device execution space.
   // Explicit annotations would trigger NVCC extended-lambda capture restrictions.
   // Vector operations.
   auto apply_system = [&](const auto &input, auto &output)
   {
      using gendil::Apply;
      Apply(system, input, output);
   };
   auto copy_vector = [&](const auto &input, auto &output)
   {
      LocalCopy(context, input, output);
   };
   auto axpy = [&](Real alpha, const auto &input, auto &output)
   {
      gendil::Axpy(context, alpha, input, output);
   };
   auto divide_vector = [&](auto &output, Real length)
   {
      LocalDivide(context, output, length);
   };
   // Reduced-system operations.
   auto store_coefficient = [&](auto &c, Integer i, Real value)
   {
      StoreCoefficient(context, c, i, value);
   };
   auto accumulate_coefficients =
       [&](auto &destination, const auto &source, Integer count)
   {
      auto x = gendil::MakeView(gendil::ReferenceContainer{source},
                                gendil::VectorLayout{count});
      auto y = gendil::MakeView(gendil::ReferenceContainer{destination},
                                gendil::VectorLayout{count});
      gendil::Axpy(context, Real(1), x, y);
   };
   auto initialize_reduced_problem = [&](Real beta)
   {
      InitializeReducedLeastSquares(context, buffers.cosines, buffers.sines,
                                    buffers.reduced_rhs, buffers.coefficients,
                                    options.restart, beta);
   };
   auto update_reduced_qr = [&](Integer j, Real next)
   {
      return UpdateReducedQR(context, buffers.hessenberg, buffers.cosines, buffers.sines,
                             buffers.reduced_rhs, j, next);
   };
   auto solve_reduced_system = [&](Integer count)
   {
      return BackSubstitute(context, buffers.hessenberg, buffers.reduced_rhs,
                            buffers.coefficients, count);
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
         LocalZero(context, buffers.work);
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
   auto compute_true_residual_norm = [&]()
   {
      solver_detail::ComputeTrueResidual(apply_system, rhs, solution, residual_vector,
                                        buffers.work, copy_vector, axpy);
      return norm(residual_vector);
   };
   auto build_arnoldi_column = [&](Integer j)
   {
      return BuildArnoldiColumn(buffers.work, j, expand_search_direction, orthogonalize,
                                norm);
   };
   const auto result = RestartedGMRES(
       options,
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
   // Complete all solver-buffer reads before return/reuse, including early failure.
   if constexpr (gendil::is_threaded_v<Context>)
      context.Synchronize();
   return result;
}

} // namespace gendil::gmres_detail
