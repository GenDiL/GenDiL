// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Solve a symmetric positive-definite system with conjugate gradients.
 *
 * Supply an operator A(input, output), a right-hand side, and an initial guess:
 *
 * @code
 * Vector x(rhs.Size());
 * x = 0.0;
 * auto result = ConjugateGradient(A, rhs, 100, Real(1e-10), x);
 * @endcode
 *
 * The short owning-vector overload allocates its work vectors. Repeated solves
 * can instead supply four reusable work vectors and a dot-product callback.
 * Fixed local storage also supports the short and KernelContext overloads.
 * All paths share the same recurrence and verify rhs - A*x before success.
 */

#include "gendil/Algebra/Vector/vector.hpp"
#include "gendil/Solvers/iterativesolverutilities.hpp"
#include "gendil/Utilities/MathHelperFunctions/dot.hpp"
#include "gendil/Utilities/MathHelperFunctions/isfinite.hpp"
#include "gendil/Utilities/MathHelperFunctions/checkedsqrt.hpp"
#include <type_traits>

namespace gendil {

/**
 * @brief Solve A*x = rhs using caller-provided work vectors and reductions.
 *
 * A must be symmetric positive definite in the inner product supplied by dot.
 * A(input, output) overwrites output; dot performs any required global or
 * collective reduction. Every norm uses this same dot, including verification
 * of the final residual. rhs need only be assignable to the work-vector type;
 * dot is called with work vectors, never directly with rhs.
 * Work vectors support value assignment, Axpy(alpha, input, output), in-place
 * scalar multiplication, and in-place addition.
 *
 * For a nonzero RHS norm, tol bounds ||rhs - A*x|| / ||rhs||. For a zero RHS
 * norm, tol bounds the absolute residual norm instead. A zero tolerance requires
 * a zero residual. Initially converged guesses and zero iteration budgets leave
 * x unchanged. There is no cutoff that treats tiny nonzero norms as zero;
 * representability of dot products remains the reduction's responsibility.
 *
 * @param[in,out] x Initial guess on entry, approximate solution on return.
 * @param[in,out] tmp,z,residual,p Four independent, preallocated work vectors.
 * @pre tol is finite and nonnegative. rhs, x, and all work vectors have matching
 *      extents and do not alias. Operations complete their outputs before use.
 *      Collective callers participate together in every operator and reduction.
 *
 * @return IterativeSolverResult with success only after true-residual
 * verification, the number of completed solution updates, and the true relative
 * residual (absolute for a zero RHS). Nonpositive curvature or invalid arithmetic
 * returns failure; an uncomputable error is infinity. Work buffers and completed
 * solution updates are not rolled back on failure.
 *
 * Verification adds an operator application at candidate convergence and at an
 * iteration-limit exit. Failed verification restarts the direction from the true
 * residual within the remaining budget; verification itself is not an iteration.
 */
template<class Operator, class RHSType, class DotType, class V>
GENDIL_HOST_DEVICE
IterativeSolverResult ConjugateGradient(
   const Operator& A, const RHSType& rhs, const DotType& dot,
   const Integer max_iters, const Real tol, V& x,
   V& tmp, V& z, V& residual, V& p)
{
   GENDIL_VERIFY(IsFinite(tol) && tol >= 0,
      "CG requires a finite, nonnegative tolerance.");

   residual = rhs;
   const Real rhs_norm = CheckedSqrt(dot(residual, residual));
   if (!solver_detail::ValidNorm(rhs_norm))
   {
      return solver_detail::MakeIterativeSolverResult(false, 0, 0, rhs_norm);
   }

   const auto copy = [](const RHSType& input, V& output) { output = input; };
   const auto axpy = [](Real alpha, const V& input, V& output)
   {
      Axpy(alpha, input, output);
   };
   auto true_residual_square = [&]()
   {
      solver_detail::ComputeTrueResidual(A, rhs, x, residual, tmp, copy, axpy);
      return Real(dot(residual, residual));
   };
   const Real absolute_tolerance = rhs_norm == 0 ? tol : 0;
   auto converged = [&](const Real norm)
   {
      return solver_detail::Converged(norm, rhs_norm, tol, absolute_tolerance);
   };
   auto result = [&](const bool success, const Integer iterations, const Real norm)
   {
      return solver_detail::MakeIterativeSolverResult(success, iterations, norm, rhs_norm);
   };
   auto failure = [&](const Integer iterations)
   {
      return result(false, iterations,
         CheckedSqrt(true_residual_square()));
   };

   Real residual_square = true_residual_square();
   Real residual_norm = CheckedSqrt(residual_square);
   if (!solver_detail::ValidNorm(residual_norm)) { return result(false, 0, residual_norm); }
   if (converged(residual_norm))
   {
      return result(true, 0, residual_norm);
   }
   if (max_iters == 0) { return result(false, 0, residual_norm); }

   p = residual;
   Integer iterations = 0;
   while (iterations < max_iters)
   {
      A(p, tmp);
      const Real curvature = dot(p, tmp);
      if (!IsFinite(curvature) || curvature <= 0) { return failure(iterations); }
      const Real alpha = residual_square / curvature;
      if (!IsFinite(alpha)) { return failure(iterations); }

      Axpy(alpha, p, x);
      Axpy(-alpha, tmp, residual);
      ++iterations;

      const Real next_square = dot(residual, residual);
      residual_norm = CheckedSqrt(next_square);
      if (!solver_detail::ValidNorm(residual_norm)) { return failure(iterations); }
      if (converged(residual_norm) || iterations == max_iters)
      {
         residual_square = true_residual_square();
         residual_norm = CheckedSqrt(residual_square);
         if (!solver_detail::ValidNorm(residual_norm))
         {
            return result(false, iterations, residual_norm);
         }
         if (converged(residual_norm))
         {
            return result(true, iterations, residual_norm);
         }
         if (iterations == max_iters) { return result(false, iterations, residual_norm); }
         // Replacing the recursive residual invalidates the old conjugate direction.
         p = residual;
         continue;
      }

      const Real beta = next_square / residual_square;
      if (!IsFinite(beta)) { return failure(iterations); }
      z = residual;
      p *= beta;
      p += z;
      residual_square = next_square;
   }
   return failure(iterations); // The loop normally returns on its last update.
}

/** @brief Solve with owning vectors, allocating four work vectors for this call.
 * This host entry point preserves the initial guess and uses Dot for reductions.
 * An empty system succeeds without allocation or operator evaluation.
 */
template<class Operator>
IterativeSolverResult ConjugateGradient(
   const Operator& A, const Vector& rhs, const Integer& max_iters,
   const Real& tol, Vector& x)
{
   GENDIL_VERIFY(IsFinite(tol) && tol >= 0,
      "CG requires a finite, nonnegative tolerance.");
   GENDIL_VERIFY(rhs.Size() == x.Size(), "CG requires matching RHS and solution sizes.");
   if (x.Size() == 0) { return {true, 0, 0}; }
   Vector tmp(x.Size()), z(x.Size()), residual(x.Size()), p(x.Size());
   const auto dot = [](const Vector& a, const Vector& b) { return Dot(a, b); };
   return ConjugateGradient(A, rhs, dot, max_iters, tol, x, tmp, z, residual, p);
}

/** @brief Solve on the host using default-constructed local work storage.
 * Default construction of V must yield independent, fully sized writable
 * storage, as it does for SerialRecursiveArray and ElementDoF. Borrowed handles
 * are not owning workspaces. Supply explicit scratch for other storage types.
 * Device callers use the Context overload or provide explicit scratch and dot.
 */
template<class Operator, class Rhs, class V>
   requires (!std::is_same_v<std::remove_cvref_t<V>, Vector>)
IterativeSolverResult ConjugateGradient(
   const Operator& A, const Rhs& rhs, const Integer& max_iters,
   const Real& tol, V& x)
{
   V tmp, z, residual, p;
   const auto dot = [](const V& a, const V& b) { return Dot(a, b); };
   return ConjugateGradient(A, rhs, dot, max_iters, tol, x, tmp, z, residual, p);
}

/** @brief Solve inside a kernel context with local storage and collective Dot.
 * Default-constructed V must own full writable storage for this work item.
 * This path allocates no dynamic vectors; ThreadedView with owning register
 * storage is supported. All participants must call it with the same budget
 * and tolerance and supply the shared memory required by contextual Dot.
 */
template<class Context, class Operator, class Rhs, class V>
GENDIL_HOST_DEVICE
IterativeSolverResult ConjugateGradient(
   Context& context, const Operator& A, const Rhs& rhs,
   const Integer& max_iters, const Real& tol, V& x)
{
   const auto dot = [&](const V& a, const V& b) { return Dot(context, a, b); };
   V tmp, z, residual, p;
   return ConjugateGradient(A, rhs, dot, max_iters, tol, x, tmp, z, residual, p);
}

} // namespace gendil
