// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Shared residual checks and reporting for iterative solvers.
 *
 * CG and GMRES use these internal helpers to form the true residual, test its
 * norm, and report the resulting error consistently. Each solver supplies its
 * own tolerances, norm operation, and iteration count. Residual formation is
 * separate from measurement so a solver can retain a squared norm without an
 * extra reduction or a square-root/square round trip.
 */
#include "gendil/Solvers/iterativesolverresult.hpp"
#include "gendil/Utilities/MathHelperFunctions/isfinite.hpp"
#include <limits>

namespace gendil::solver_detail
{

/// Accept a finite, nonnegative result from a supplied norm operation.
GENDIL_HOST_DEVICE inline bool ValidNorm(const Real norm)
{
   return IsFinite(norm) && norm >= 0;
}

/** Test a residual against independent relative and absolute tolerances.
 * Tolerances must be finite and nonnegative. For a zero RHS only the absolute
 * tolerance applies. The positive relative-tolerance guard prevents an
 * underflowed quotient from satisfying a request for an exactly zero residual.
 */
GENDIL_HOST_DEVICE inline bool Converged(
   const Real residual, const Real rhs,
   const Real relative_tolerance, const Real absolute_tolerance)
{
   return ValidNorm(residual) && ValidNorm(rhs) &&
      (residual <= absolute_tolerance ||
       (rhs > 0 && relative_tolerance > 0 && residual / rhs <= relative_tolerance));
}

/** Report relative residual for nonzero RHS norm, otherwise absolute residual.
 * Invalid norms force failure with infinite error. A finite norm pair whose
 * quotient overflows reports infinity while preserving the supplied success
 * flag: the solver may have satisfied an absolute tolerance. Iteration-count
 * semantics remain the caller's responsibility.
 */
GENDIL_HOST_DEVICE inline IterativeSolverResult MakeIterativeSolverResult(
   const bool success, const Integer iterations, const Real residual, const Real rhs)
{
   const Real infinity = std::numeric_limits<Real>::infinity();
   if (!ValidNorm(residual) || !ValidNorm(rhs)) { return {false, iterations, infinity}; }
   const Real error = rhs > 0 ? residual / rhs : residual;
   return {success, iterations, IsFinite(error) ? error : infinity};
}

/** Form residual = rhs - A*x, using work for A*x, without measuring its norm.
 * Callbacks are borrowed, including mutable/noncopyable objects. apply must
 * preserve x and overwrite work; copy and axpy update mathematical entries in
 * the preallocated residual. The supplied operands must have matching extents
 * and independent storage. Each operation completes its output before the next;
 * this helper adds no synchronization, allocation, or reduction.
 */
template<class Apply, class Rhs, class X, class Residual, class Work,
         class Copy, class Axpy>
GENDIL_HOST_DEVICE void ComputeTrueResidual(
   Apply& apply, const Rhs& rhs, X& x, Residual& residual, Work& work,
   Copy& copy, Axpy& axpy)
{
   apply(x, work);
   copy(rhs, residual);
   axpy(Real(-1), work, residual);
}

} // namespace gendil::solver_detail
