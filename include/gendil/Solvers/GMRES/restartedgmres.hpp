// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Repeat bounded Krylov searches until the original system is solved.
 *
 * A GMRES cycle builds search directions, solves a small least-squares problem,
 * and updates the solution. Restarting begins another cycle from the new
 * residual while limiting the number of retained directions. The norm of
 * rhs - A*solution, not the reduced problem's estimate, determines success.
 *
 * This is the shared numerical controller used by the public GMRES entry
 * points. Supplied operations perform residual evaluation, Arnoldi expansion,
 * reduced solves, and corrections; this header owns their order, the iteration
 * budget, and the restart decision. It neither chooses execution nor interprets
 * dense scratch layouts.
 */
#include "gendil/Solvers/GMRES/arnoldi.hpp"
#include "gendil/Solvers/iterativesolverresult.hpp"
#include <limits>

namespace gendil::gmres_detail
{
/** Form residual = rhs - A*x, using work for A*x, and return its norm.
 * The solution is unchanged. Each supplied operation completes the output
 * consumed by the next; this helper adds no collective synchronization.
 */
template <class Apply, class Rhs, class X, class Residual, class Work, class Copy,
          class Axpy, class Norm>
GENDIL_HOST_DEVICE Real ComputeTrueResidualNorm(Apply &apply, const Rhs &rhs, X &x,
                                                Residual &residual, Work &work,
                                                Copy &copy, Axpy &axpy, Norm &norm)
{
   apply(x, work);
   copy(rhs, residual);
   axpy(Real(-1), work, residual);
   return norm(residual);
}

/// Test finite norms against either tolerance, using division to avoid a tolerance-product overflow.
GENDIL_HOST_DEVICE inline bool Converged(Real residual, Real rhs,
                                         const GMRESOptions &options)
{
   return ValidNorm(residual) && ValidNorm(rhs) &&
          (residual <= options.absolute_tolerance ||
           (rhs > 0 && options.relative_tolerance > 0 &&
            residual / rhs <= options.relative_tolerance));
}

/** Report relative residual for nonzero RHS norm, otherwise absolute residual.
 * Invalid norms force failure with infinite error. An unrepresentable relative
 * quotient is also reported as infinity without changing the supplied success flag.
 */
GENDIL_HOST_DEVICE inline IterativeSolverResult
MakeGMRESResult(bool success, Integer iterations, Real residual, Real rhs)
{
   const Real infinity = std::numeric_limits<Real>::infinity();
   if (!ValidNorm(residual) || !ValidNorm(rhs))
      return {false, iterations, infinity};
   const Real error = rhs > 0 ? residual / rhs : residual;
   return {success, iterations, IsFinite(error) ? error : infinity};
}

/** Run restart cycles using already-bound numerical operations and scratch.
 *
 * compute_true_residual_norm() must write the residual into buffers.basis(0)
 * as well as return its norm. build_arnoldi_column(j) writes coefficients through
 * its bindings and leaves the unnormalized remainder in buffers.work.
 * The controller normalizes basis vectors, initializes each reduced problem,
 * reduces columns, and requests a correction only after a successful back solve.
 *
 * rhs_norm is the supplied norm of the original RHS. Norms and projections must
 * describe the same geometry. Scratch dimensions and collective participation
 * are entry-point preconditions; callbacks are borrowed and complete their
 * advertised outputs before returning. No execution context or barrier is
 * introduced here.
 *
 * Iterations count attempted Arnoldi columns, including failed ones. A dependent
 * direction can still yield a usable correction; a singular new reduced column
 * permits a correction from previously accepted columns. Nonfinite columns or
 * a failed back solve skip the incomplete cycle's correction. Existing solution
 * updates from earlier cycles and scratch writes are not rolled back.
 *
 * Recompute the true residual after every correction and before reporting a
 * failed cycle. Zero budget or zero restart evaluates the initial residual
 * without changing the solution. Reduced residual estimates may shorten a
 * cycle but cannot establish final convergence.
 */
template <class Buffers, class Residual, class Build, class Correct, class Copy,
          class Divide, class Initialize, class Reduce, class Solve>
GENDIL_HOST_DEVICE IterativeSolverResult
RestartedGMRES(const GMRESOptions &options, Buffers &buffers, Real rhs_norm,
               Residual &compute_true_residual_norm, Build &build_arnoldi_column,
               Correct &apply_correction, Copy &copy_vector, Divide &divide_vector,
               Initialize &initialize_reduced_problem, Reduce &update_reduced_qr,
               Solve &solve_reduced_system)
{
   Real beta = compute_true_residual_norm();
   if (!ValidNorm(rhs_norm) || !ValidNorm(beta))
      return MakeGMRESResult(false, 0, beta, rhs_norm);
   if (Converged(beta, rhs_norm, options))
      return MakeGMRESResult(true, 0, beta, rhs_norm);
   if (options.max_iterations == 0 || options.restart == 0)
      return MakeGMRESResult(false, 0, beta, rhs_norm);

   Integer iterations = 0;
   constexpr Real epsilon = std::numeric_limits<Real>::epsilon();
   while (iterations < options.max_iterations)
   {
      auto &&first = buffers.basis(0);
      divide_vector(first, beta);
      initialize_reduced_problem(beta);
      Integer accepted = 0;
      bool dependent = false, failure = false;
      for (Integer j = 0; j < options.restart && iterations < options.max_iterations; ++j)
      {
         const ArnoldiColumnResult column = build_arnoldi_column(j);
         ++iterations;
         if (!column.valid)
         {
            failure = true;
            break;
         }
         const auto reduced = update_reduced_qr(j, column.orthogonal_remainder_norm);
         if (reduced.status == decltype(reduced.status)::NonFinite)
         {
            failure = true;
            break;
         }
         if (reduced.status == decltype(reduced.status)::Singular)
         {
            dependent = true;
            break;
         }
         accepted = j + 1;
         dependent = column.orthogonal_remainder_norm <=
                     epsilon * column.norm_before_orthogonalization;
         if (dependent || Converged(reduced.residual_norm_estimate, rhs_norm, options))
            break;
         if (j + 1 < options.restart && iterations < options.max_iterations)
         {
            auto &&next = buffers.basis(j + 1);
            copy_vector(buffers.work, next);
            divide_vector(next, column.orthogonal_remainder_norm);
         }
      }
      if (failure || accepted == 0 || !solve_reduced_system(accepted))
      {
         beta = compute_true_residual_norm();
         return MakeGMRESResult(false, iterations, beta, rhs_norm);
      }
      apply_correction(accepted);
      beta = compute_true_residual_norm();
      const bool success = Converged(beta, rhs_norm, options);
      if (success || dependent || !ValidNorm(beta))
         return MakeGMRESResult(success, iterations, beta, rhs_norm);
   }
   return MakeGMRESResult(false, iterations, beta, rhs_norm);
}

} // namespace gendil::gmres_detail
