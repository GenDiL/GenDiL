// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Minimal, uniform return type for iterative linear solvers.
 *
 * This header defines @ref gendil::IterativeSolverResult, a light POD used by
 * GenDiL solvers (e.g., CG, GMRES, BiCGStab) to report convergence status,
 * iteration count, and a normalized error metric.
 *
 */

#include <gendil/prelude.hpp>

namespace gendil {

/**
 * @brief Outcome of an iterative solve.
 *
 * A small, trivially copyable record returned by iterative methods to convey
 * whether stopping criteria were met, how many iterations were performed, and
 * the final value of the solver's monitored relative error.
 *
 * @par Field semantics
 *  - `success` :
 *      `true` if the solver met its stopping criterion (e.g., tolerance or
 *      stagnation rule), `false` otherwise (e.g., max iterations reached).
 *  - `iterations` :
 *      Number of *completed* iterations. Zero means the initial guess already
 *      satisfied the stopping criterion.
 *  - `relative_error` :
 *      Final normalized error/residual reported by the algorithm
 *      (e.g., ‖r_k‖/‖b‖ or ‖r_k‖/‖r_0‖). The exact normalization is
 *      solver-defined and should be documented at the call site.
 *
 */
struct IterativeSolverResult
{
   bool success;
   Integer iterations;
   Real relative_error;
};

}
