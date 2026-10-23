// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Repeat a bound iterative solve from its latest approximation. */

#include "gendil/Solvers/iterativesolverresult.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include <concepts>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * A bound solver starts from initial, writes the reached approximation to
 * solution, and returns the outcome of that invocation. Distinct input storage
 * is preserved. The output and result are defined on unsuccessful exits too.
 *
 * This concept checks the callable signature, including mutable callables.
 * Storage ownership, alias safety, convergence criteria, and execution context
 * are contracts of the particular solver, not requirements checked here.
 */
template<class Solver, class InputVector, class OutputVector = InputVector>
concept InitializedIterativeSolver = requires(
   Solver& solver, const InputVector& initial, OutputVector& solution)
{
   { solver(initial, solution) } -> std::same_as<IterativeSolverResult>;
};

/**
 * @brief Bind a solver to a finite restart allowance and an explicit retry decision.
 *
 * The returned host callable has the same (initial, solution) interface as
 * solve. For example, a caller with solver-specific recovery information can use:
 * @code
 * auto restarted = RestartedSolver(solve, 3, should_restart);
 * auto result = restarted(initial, solution);
 * @endcode
 *
 * The initial call always runs. After an unsuccessful call, should_restart is
 * consulted only if another restart is available. It receives that attempt's
 * result and must identify which failures permit recovery; a finite error alone
 * does not distinguish breakdown from invalid arithmetic. Success stops
 * immediately. Zero max_restarts allows only the initial call.
 *
 * Each retry invokes solve(solution, solution), so solve must support exact
 * input/output aliasing and initialize a fresh recurrence on every invocation.
 * This avoids imposing vector allocation or copying on the controller. Partial
 * overlap is not supported unless the bound solver explicitly permits it.
 * Distinct initial input is preserved. Captured storage must outlive all uses.
 *
 * The result retains the last attempt's success and error and sums all reported
 * iterations (the sum must be representable). There is no total iteration
 * budget or independent convergence test. All attempts use the bound solver's
 * convergence contract. Invalid output is not rolled back; the retry decision
 * must prevent its reuse unless recovery is explicitly supported by solve.
 *
 * Callables are stored by value, supporting moves and mutable state; use
 * std::ref to borrow them. Restart/iteration counters reset on each invocation,
 * while state owned by the supplied callables follows their own contracts.
 */
template<class Solver, class ShouldRestart>
auto RestartedSolver(Solver&& solve, Integer max_restarts, ShouldRestart&& should_restart)
{
   return [solve = std::forward<Solver>(solve), max_restarts,
           should_restart = std::forward<ShouldRestart>(should_restart)]
      <class Input, class Output>(const Input& initial, Output& solution) mutable
      -> IterativeSolverResult
      requires (
         InitializedIterativeSolver<std::decay_t<Solver>, Input, Output> &&
         InitializedIterativeSolver<std::decay_t<Solver>, Output> &&
         requires(std::decay_t<ShouldRestart>& retry, const IterativeSolverResult& latest)
         {
            { retry(latest) } -> std::convertible_to<bool>;
         })
   {
      auto result = solve(initial, solution);
      Integer iterations = result.iterations;
      Integer restarts = 0;
      while (!result.success && restarts < max_restarts &&
             should_restart(std::as_const(result)))
      {
         ++restarts;
         result = solve(std::as_const(solution), solution);
         iterations = CheckedAdd(iterations, result.iterations,
            "Restarted solver iteration count overflow.");
      }
      result.iterations = iterations;
      return result;
   };
}

} // namespace gendil
