// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <limits>
#include <tuple>
#include "gendil/prelude.hpp"

namespace gendil {

/**
 * @brief Solve A x = rhs using (restarted) GMRES without any internal allocations.
 *
 * This version requires the caller to hand in *all* needed scratch space, including
 * a length‐`restart` array `y` for the small least‐squares solve.  No `resize` or
 * `new` happens inside.
 *
 * @tparam Operator   Functor for matrix‐vector product: w = A * v
 * @tparam DotType    Dot‐product functor: returns Real dot(const Vector&, const Vector&)
 * @tparam Vector     Mutable vector type (e.g. std::vector<Real> or a GPU vector)
 *
 * @param[in]     A            Linear operator: A(v, w) ⟹ w = A·v
 * @param[in]     rhs          Right‐hand‐side vector
 * @param[in]     dot          Dot‐product function (performs any global reductions)
 * @param[in]     max_iters    Maximum total GMRES iterations (counting restarts)
 * @param[in]     restart      GMRES restart length (m).  Typically m ≪ dimension.
 * @param[in]     tol          Convergence tolerance (relative to ‖rhs‖)
 * @param[in,out] x            On entry: initial guess. On exit: approximate solution.
 *
 * @param[in,out] V            Pre‐allocated array of (restart+1) Vectors,
 *                             each of size = x.size().  Used to store Krylov basis.
 * @param[in,out] w            Pre‐allocated workspace Vector (size = x.size()),
 *                             used for computing A·v_j.
 * @param[in,out] hessenberg   Pre‐allocated flat array of length (restart+1)*restart,
 *                             storing H in column-major order; i.e.
 *                             H(i,j) = hessenberg[j*(restart+1) + i].
 * @param[in,out] cs           Pre-allocated array of length restart for Givens cosines.
 * @param[in,out] sn           Pre-allocated array of length restart for Givens sines.
 * @param[in,out] e1_rhs       Pre-allocated array of length (restart+1) for the small LS RHS.
 * @param[in,out] y            Pre-allocated array of length restart for the small dense unknowns.
 *
 * @return std::tuple<bool, Integer, Real>
 *         - bool:    true if converged (relative residual ≤ tol)
 *         - Integer: total number of GMRES iterations performed
 *         - Real:    final relative residual norm (sqrt(‖r‖²/‖rhs‖²))
 */
template<
    typename Operator,
    typename RHSType,
    typename DotType,
    typename VectorArray,
    typename Vector,
    typename Array
>
GENDIL_HOST_DEVICE
auto GMRES_no_alloc(
    const Operator &         A,
    const RHSType &          rhs,
    const DotType &          dot,
    const Integer            max_iters,
    const Integer            restart,
    const Real               tol,
    Vector &                 x,
    VectorArray &            V,           // array of (restart+1) Vectors
    Vector &                 w,           // single workspace Vector
    Array &                  hessenberg,  // length = (restart+1)*restart
    Array &                  cs,          // length = restart
    Array &                  sn,          // length = restart
    Array &                  e1_rhs,      // length = restart+1
    Array &                  y            // length = restart
)
{
    using Result = std::tuple<bool, Integer, Real>;
    const Real tol_rel = tol;
    const Real tiny    = std::numeric_limits<Real>::epsilon();

    // 1) Compute initial residual r0 = rhs - A*x; store in V[0]
    A(x, w);
    V[0] = rhs;
    V[0] -= w;
    Real beta = Sqrt(dot(V[0], V[0]));
    if (beta == Real(0))
    {
        // Exact solution already
        return Result{ true, 0, Real(0) };
    }

    // 2) Compute ‖rhs‖ for relative tol
    Real norm_rhs = Sqrt(dot(rhs, rhs));
    if (norm_rhs == Real(0))
    {
        x = Real(0);
        return Result{ true, 0, Real(0) };
    }

    Integer total_iters = 0;

    // Outer restart loop
    while (total_iters < max_iters)
    {
        bool inner_converged = false;

        // Reset Givens arrays
        for (Integer i = 0; i < restart; ++i)
        {
            cs[i] = Real(1);
            sn[i] = Real(0);
        }

        // Normalize v0 = r0 / beta
        const Real inv_beta = Real(1) / beta;
        V[0] *= inv_beta;

        // Initialize small‐RHS for least‐squares: e1_rhs = [beta, 0, 0, ..., 0]^T
        for (Integer i = 0; i <= restart; ++i) e1_rhs[i] = Real(0);
        e1_rhs[0] = beta;

        Integer m = 0;  // current Arnoldi column (0-based)

        // Inner Arnoldi loop (at most 'restart' steps, and ≤ max_iters total)
        for ( ; m < restart && total_iters < max_iters; ++m, ++total_iters)
        {
            // 3) w = A * v_m
            A(V[m], w);

            // 4) Modified Gram–Schmidt: orthonormalize w against V[0..m]
            for (Integer i = 0; i <= m; ++i)
            {
                Real hij = dot(V[i], w);
                hessenberg[m*(restart+1) + i] = hij;   // H(i,m)
                Axpy(-hij, V[i], w);
            }

            Real h_next = Sqrt(dot(w, w));
            hessenberg[m*(restart+1) + (m+1)] = h_next;  // H(m+1,m)
            bool happy = (h_next <= tiny * norm_rhs);

            // 5) Apply all previous Givens rotations to H[0..m+1, m]
            for (Integer i = 0; i < m; ++i)
            {
                Real h_im   = hessenberg[m*(restart+1) + i];
                Real h_ip1m = hessenberg[m*(restart+1) + (i+1)];
                Real temp1  = cs[i] * h_im + sn[i] * h_ip1m;
                Real temp2  = -sn[i] * h_im + cs[i] * h_ip1m;
                hessenberg[m*(restart+1) + i]     = temp1;  // H(i,m)
                hessenberg[m*(restart+1) + (i+1)] = temp2;  // H(i+1,m)
            }

            // 6) Generate new Givens to zero H(m+1,m)
            {
                Real h_mm   = hessenberg[m*(restart+1) + m];
                Real h_m1m  = hessenberg[m*(restart+1) + (m+1)];
                Real rho    = Sqrt(h_mm*h_mm + h_m1m*h_m1m);
                if (rho <= tiny) { cs[m] = Real(1); sn[m] = Real(0); }
                else             { cs[m] =  h_mm / rho; sn[m] =  h_m1m / rho; }

                // Rotate into H(m,m) and H(m+1,m)
                hessenberg[m*(restart+1) + m]     = cs[m] * h_mm + sn[m] * h_m1m;
                hessenberg[m*(restart+1) + (m+1)] = Real(0);

                // Also rotate the small RHS entries e1_rhs[m], e1_rhs[m+1]
                Real g_m   = e1_rhs[m];
                Real g_m1  = e1_rhs[m+1];
                e1_rhs[m]   =  cs[m] * g_m  + sn[m] * g_m1;
                e1_rhs[m+1] = -sn[m] * g_m  + cs[m] * g_m1;
            }

            // 7) Check convergence:  |e1_rhs[m+1]| / norm_rhs < tol_rel?
            Real rel_res = Abs(e1_rhs[m+1]) / norm_rhs;
            if (rel_res <= tol_rel)
            {
                inner_converged = true;   // converged using up to column m+1
                break;
            }

            // 8) Normalize v_{m+1} only if we continue
            if (!happy)
            {
                V[m+1] = w;
                V[m+1] *= (Real(1) / h_next);
            }
        }

        // Use k = actual Krylov dimension: m if full cycle, else m+1
        Integer k = inner_converged ? (m + 1) : m;

        // Back‐substitution on the k×k upper‐triangular leading block of H:
        for (Integer i = k;  i-- > 0; )
        {
            Real sum = e1_rhs[i];
            for (Integer j = i+1;  j < k;  ++j)
            {
                // H(i,j) is stored at hessenberg[j*(restart+1) + i]
                sum -= hessenberg[j*(restart+1) + i] * y[j];
            }
            y[i] = sum / hessenberg[i*(restart+1) + i]; // divide by R(i,i)
        }

        // 9) Update solution:  x += Σ_{i=0..(k-1)} y[i] * V[i]
        for (Integer i = 0; i < k; ++i)
        {
            Axpy(y[i], V[i], x);
        }

        // 10) Compute final relative residual = |e1_rhs[k]| / norm_rhs
        Real final_rel_res = Abs(e1_rhs[k]) / norm_rhs;
        if (final_rel_res <= tol_rel)
        {
            // Converged within this cycle
            return Result{ true, total_iters, final_rel_res };
        }

        // 11) If we still have iterations left, form new residual r0 = rhs - A*x
        if (total_iters >= max_iters)
        {
            // Out of total iterations
            return Result{ false, total_iters, final_rel_res };
        }

        // Compute new r0 → store in V[0], and continue with next restart cycle
        A(x, w);
        V[0] = rhs;
        V[0] -= w;
        beta = Sqrt(dot(V[0], V[0]));
        if (beta == Real(0))
        {
            // Exact solution reached after an outer cycle
            return Result{ true, total_iters, Real(0) };
        }
        // Next cycle will renormalize: V[0] *= 1/beta and reset e1_rhs, cs, sn ...
    }

    // If we reach here, max_iters was exhausted.  Recompute actual residual:
    A(x, w);
    V[0] = rhs;
    V[0] -= w;
    Real rnorm = Sqrt(dot(V[0], V[0]));
    return Result{ false, total_iters, rnorm / Sqrt(dot(rhs, rhs)) };
}

} // namespace gendil
