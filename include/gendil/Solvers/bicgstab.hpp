// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

/**
 * @brief Unpreconditioned BiCGSTAB with external allocations
 *
 * All workspace must be preallocated.
 *
 * @tparam Operator Matrix-free linear operator: A(v, w) sets w = A*v
 * @tparam Dot      Dot-product functor: returns Real dot(u, v)
 * @tparam Vector   Vector type with basic operations
 *
 * @param[in]     A         Operator: A(v, w) ⟹ w = A·v
 * @param[in]     rhs       Right-hand-side vector
 * @param[in]     dot       Dot product function (for reductions)
 * @param[in]     max_iter  Maximum number of iterations
 * @param[in]     tol       Relative tolerance on residual norm
 * @param[in,out] x         On entry: initial guess; On exit: solution
 *
 * @param[in,out] r, r_hat, p, v, s, t: all preallocated and sized to match x
 *
 * @return std::tuple<bool, Integer, Real>
 *         - bool:    true if converged
 *         - Integer: number of iterations performed
 *         - Real:    final relative residual
 */
template <
    typename Operator,
    typename RHSType,
    typename Dot,
    typename Vector,
    typename Real = gendil::Real
>
GENDIL_HOST_DEVICE
auto BiCGSTAB_no_alloc(
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
    using Result = std::tuple<bool, Integer, Real>;

    // Initial residual: r = rhs - A*x
    A(x, t);     // t = A*x
    r = rhs;
    r -= t;

    r_hat = r;

    Real norm_rhs = Sqrt(dot(rhs, rhs));
    if (norm_rhs == 0.0)
    {
        x = 0.0;
        return Result{true, 0, 0.0};
    }

    Real rho_prev = 1.0;
    Real alpha = 1.0;
    Real omega = 1.0;

    p = 0.0;
    v = 0.0;

    Real rnorm = Sqrt(dot(r, r));
    Real rel_res = rnorm / norm_rhs;
    if (rel_res < tol)
    {
        return Result{true, 0, rel_res};
    }

    Integer iter = 0;
    for (; iter < max_iter; ++iter)
    {
        Real rho = dot(r_hat, r);
        if (Abs(rho) < 1e-20) break;

        if (iter == 0)
        {
            p = r;
        }
        else
        {
            Real beta = (rho / rho_prev) * (alpha / omega);
            Axpy(-omega, v, p);   // p = p - omega * v
            p *= beta;            // p = beta * (p - omega * v)
            Axpy(1.0, r, p);      // p = r + ...
        }

        A(p, v);
        Real rhat_dot_v = dot(r_hat, v);
        if (Abs(rhat_dot_v) < 1e-20) break;

        alpha = rho / rhat_dot_v;

        s = r;
        Axpy(-alpha, v, s);  // s = r - alpha * v

        A(s, t);             // t = A * s
        Real t_dot_s = dot(t, s);
        Real t_dot_t = dot(t, t);

        if (Abs(t_dot_t) < 1e-20) break;

        omega = t_dot_s / t_dot_t;

        Axpy(alpha, p, x);  // x += alpha * p
        Axpy(omega, s, x);  // x += omega * s

        r = s;
        Axpy(-omega, t, r); // r = s - omega * t

        rnorm = Sqrt(dot(r, r));
        rel_res = rnorm / norm_rhs;

        if (rel_res < tol)
        {
            return Result{true, iter + 1, rel_res};
        }

        rho_prev = rho;
    }

    return Result{false, iter, rel_res};
}

} // namespace gendil
