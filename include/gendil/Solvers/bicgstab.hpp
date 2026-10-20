// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Solve a nonsymmetric linear system with ordinary or explicitly restarted BiCGSTAB. */

#include "gendil/Solvers/iterativesolverutilities.hpp"
#include "gendil/Solvers/restartedsolver.hpp"
#include "gendil/Utilities/MathHelperFunctions/checkedsqrt.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil {

namespace solver_detail {

// Recovery information stays private until the shared result diagnostics are
// designed. A recomputable finite residual does not make invalid arithmetic
// recoverable.
struct BiCGSTABOutcome
{
    IterativeSolverResult result;
    bool can_restart = false;
};

/** Run one fresh recurrence, verifying its residual before returning. */
template <
    typename Operator,
    typename RHSType,
    typename Dot,
    typename Vector,
    typename Real = gendil::Real
>
BiCGSTABOutcome BiCGSTAB(
    const Operator& A,
    const RHSType& rhs,
    const Dot& dot,
    Integer max_iter,
    Real tol,
    Vector& x,
    Vector& r, Vector& r_hat,
    Vector& p, Vector& v,
    Vector& s, Vector& t)
{
    r = rhs;
    const Real norm_rhs = CheckedSqrt(dot(r, r));
    if (!solver_detail::ValidNorm(norm_rhs))
    {
        return {solver_detail::MakeIterativeSolverResult(false, 0, 0, norm_rhs)};
    }

    const auto copy = [](const RHSType& input, Vector& output) { output = input; };
    const auto axpy = [](Real coefficient, const Vector& input, Vector& output)
    {
        Axpy(coefficient, input, output);
    };
    auto true_residual_norm = [&]()
    {
        solver_detail::ComputeTrueResidual(A, rhs, x, r, t, copy, axpy);
        return Real(CheckedSqrt(dot(r, r)));
    };
    const Real absolute_tolerance = norm_rhs == 0 ? tol : 0;
    auto converged = [&](const Real norm)
    {
        return solver_detail::Converged(norm, norm_rhs, tol, absolute_tolerance);
    };

    Integer iterations = 0;
    Real residual_norm = true_residual_norm();
    auto result = [&](const bool success)
    {
        return solver_detail::MakeIterativeSolverResult(
            success, iterations, residual_norm, norm_rhs);
    };
    auto failure = [&]()
    {
        residual_norm = true_residual_norm();
        return BiCGSTABOutcome{result(false)};
    };

    if (!solver_detail::ValidNorm(residual_norm)) { return failure(); }
    if (converged(residual_norm)) { return {result(true)}; }
    if (max_iter == 0) { return {result(false)}; }

    r_hat = r;
    Real rho_prev = Real(1);
    Real alpha = Real(1);
    Real omega = Real(1);
    bool fresh = true;

    const auto verify = [&]()
    {
        residual_norm = true_residual_norm();
        const bool success = converged(residual_norm);
        return BiCGSTABOutcome{
            result(success), solver_detail::ValidNorm(residual_norm) &&
                             !success && iterations < max_iter};
    };

    while (iterations < max_iter)
    {
        const Real rho = dot(r_hat, r);
        if (!IsFinite(rho)) { return failure(); }
        if (rho == 0)
        {
            return verify();
        }

        if (fresh)
        {
            p = r;
            fresh = false;
        }
        else
        {
            const Real beta = (rho / rho_prev) * (alpha / omega);
            if (!IsFinite(beta)) { return failure(); }
            Axpy(-omega, v, p);
            p *= beta;
            Axpy(Real(1), r, p);
        }

        A(p, v);
        const Real rhat_dot_v = dot(r_hat, v);
        if (!IsFinite(rhat_dot_v)) { return failure(); }
        if (rhat_dot_v == 0)
        {
            return verify();
        }

        alpha = rho / rhat_dot_v;
        if (!IsFinite(alpha)) { return failure(); }
        s = r;
        Axpy(-alpha, v, s);
        const Real snorm = CheckedSqrt(dot(s, s));
        if (!solver_detail::ValidNorm(snorm)) { return failure(); }
        if (converged(snorm))
        {
            Axpy(alpha, p, x);
            ++iterations;
            return verify();
        }

        A(s, t);
        const Real t_dot_s = dot(t, s);
        const Real t_dot_t = dot(t, t);
        if (!IsFinite(t_dot_s) || !solver_detail::ValidNorm(t_dot_t))
        {
            return failure();
        }

        // A zero stabilization denominator or coefficient retains alpha progress.
        // Invalid stabilization arithmetic leaves that pending update uncommitted.
        omega = t_dot_t == 0 ? Real(0) : t_dot_s / t_dot_t;
        if (!IsFinite(omega)) { return failure(); }
        Axpy(alpha, p, x);
        if (omega == 0)
        {
            ++iterations;
            return verify();
        }

        Axpy(omega, s, x);
        ++iterations;
        r = s;
        Axpy(-omega, t, r);
        residual_norm = CheckedSqrt(dot(r, r));
        if (!solver_detail::ValidNorm(residual_norm)) { return failure(); }
        if (converged(residual_norm) || iterations == max_iter)
        {
            return verify();
        }
        rho_prev = rho;
    }
    return failure(); // The loop normally returns on its last completed step.
}

} // namespace solver_detail

/**
 * @brief Solve A*x = rhs with one BiCGSTAB recurrence and caller-owned workspace.
 *
 * Supply an operator A(input, output) that overwrites output and a dot product
 * defined on the work-vector type. rhs need only be assignable to that type.
 * Work vectors support value assignment, Axpy, and in-place scalar multiplication.
 * This host entry point allocates no vectors. Use RestartedBiCGSTAB_no_alloc
 * with an explicit restart limit when recovery from finite breakdown is wanted.
 *
 * For nonzero RHS norm, tol bounds ||rhs - A*x||/||rhs||. For zero RHS norm it
 * bounds the absolute residual instead. Comparisons are inclusive, and zero
 * tolerance requires a zero residual. Initially converged guesses and zero
 * iteration budgets preserve x. Every norm uses the supplied dot product;
 * RHS normalization uses a workspace copy. Convergence and terminal residual
 * reports are checked against rhs - A*x.
 *
 * Finite breakdown or rejected convergence ends this attempt after residual
 * verification. A complete alpha/omega step counts once, as does an alpha-only
 * update retained when its residual converges or finite stabilization breaks
 * down. Verification does not consume iterations. Negative recurrence
 * coefficients are valid. Invalid arithmetic always terminates with failure,
 * without committing a pending update known to be invalid; completed updates
 * are not rolled back.
 *
 * @param[in,out] x Initial guess on entry, approximate solution on return.
 * @param[in,out] r,r_hat,p,v,s,t Six preallocated work vectors matching x.
 * @pre tol is finite and nonnegative. Operands/workspaces have matching extents
 *      and independent storage. Operations complete their outputs before use.
 * @return IterativeSolverResult with verified success, completed step count,
 *         and true relative residual (absolute for zero RHS). An uncomputable
 *         error is infinity. Extreme reduction overflow/underflow is governed
 *         by the supplied dot product.
 */
template <
    typename Operator,
    typename RHSType,
    typename Dot,
    typename Vector,
    typename Real = gendil::Real
>
IterativeSolverResult BiCGSTAB(
    const Operator& A,
    const RHSType& rhs,
    const Dot& dot,
    Integer max_iter,
    Real tol,
    Vector& x,
    Vector& r, Vector& r_hat,
    Vector& p, Vector& v,
    Vector& s, Vector& t)
{
    GENDIL_VERIFY(IsFinite(tol) && tol >= 0,
        "BiCGSTAB requires a finite, nonnegative tolerance.");
    return solver_detail::BiCGSTAB(
        A, rhs, dot, max_iter, tol, x, r, r_hat, p, v, s, t).result;
}

/**
 * @brief Restart BiCGSTAB from its latest approximation, with an explicit limit.
 *
 * Finite breakdown or rejected convergence can start a fresh recurrence.
 * max_restarts counts all restarts across this call; the initial attempt is
 * free and zero disables recovery. max_iter bounds the total completed steps
 * across all attempts. Invalid arithmetic is never retried.
 *
 * Each fresh attempt recomputes the RHS norm and initial true residual, including
 * after the preceding attempt verified its final residual. All attempts reuse
 * the same six work vectors without allocating vectors. Recovery does not
 * guarantee convergence. Convergence, reporting, and operand requirements are
 * those of BiCGSTAB. There is no implicit restart limit.
 */
template <
    typename Operator,
    typename RHSType,
    typename Dot,
    typename Vector,
    typename Real = gendil::Real
>
IterativeSolverResult RestartedBiCGSTAB_no_alloc(
    const Operator& A,
    const RHSType& rhs,
    const Dot& dot,
    Integer max_iter,
    Real tol,
    Vector& x,
    Vector& r, Vector& r_hat,
    Vector& p, Vector& v,
    Vector& s, Vector& t,
    Integer max_restarts)
{
    GENDIL_VERIFY(IsFinite(tol) && tol >= 0,
        "BiCGSTAB requires a finite, nonnegative tolerance.");

    Integer remaining = max_iter;
    bool can_restart = false;
    auto attempt = [&](const Vector& initial, Vector& solution)
    {
        if (&initial != &solution) { solution = initial; }
        can_restart = false;
        const auto outcome = solver_detail::BiCGSTAB(
            A, rhs, dot, remaining, tol, solution, r, r_hat, p, v, s, t);
        remaining -= outcome.result.iterations;
        can_restart = outcome.can_restart;
        return outcome.result;
    };
    const auto should_restart = [&](const IterativeSolverResult&)
    {
        return can_restart && remaining > 0;
    };
    auto restarted = RestartedSolver(attempt, max_restarts, should_restart);
    return restarted(x, x);
}

} // namespace gendil
