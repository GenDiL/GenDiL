// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Choose GMRES accuracy and the amount of iteration work allowed. */

#include <cmath>
#include <limits>
#include "gendil/Utilities/types.hpp"

namespace gendil {

/**
 * @brief Set how much work GMRES may do and how accurately it must solve A x = rhs.
 *
 * Set named fields so the restart length and the two tolerances are easy to
 * distinguish at the call site. For an existing map A, right-hand side rhs,
 * and initial guess solution, include @c gendil/Solvers/GMRES/gmres.hpp and use:
 * @code
 * GMRESOptions options;
 * options.restart = 20;
 * options.max_iterations = 200;
 * options.relative_tolerance = 1.e-10;
 * options.absolute_tolerance = 1.e-14;
 * auto result = GMRES(A, rhs, options, solution);
 * @endcode
 * The call updates solution and reports convergence in result.
 *
 * Options are read for this synchronous solve. Change them between solves;
 * reusable-workspace overloads use the same settings without modifying them.
 *
 * Stopping uses the true residual rhs - A*solution in the solve's selected
 * norm. Either tolerance can accept the residual; for a zero right-hand side,
 * only the absolute tolerance applies. Tolerances must be finite and
 * nonnegative. Setting both to zero requests an exactly zero residual.
 */
struct GMRESOptions
{
   /// Maximum Arnoldi steps per cycle; must be positive. Larger values retain
   /// more search directions but require more storage and orthogonalization.
   /// Host-driven solves cap this at the vector length; local solves require
   /// it to fit the caller's workspace capacity.
   Integer restart = 30;
   /// Total attempted Arnoldi-step budget across all cycles, not the number of restarts.
   /// Zero checks the initial residual without changing the solution; an empty
   /// host system returns success without numerical callbacks.
   Integer max_iterations = 300;
   /// Dimensionless bound on norm(rhs - A*solution) / norm(rhs) for nonzero rhs.
   Real relative_tolerance =
      std::sqrt(std::numeric_limits<Real>::epsilon());
   /// Bound on norm(rhs - A*solution), in residual-norm units; useful near zero rhs.
   Real absolute_tolerance = 0.0;
};

} // namespace gendil
