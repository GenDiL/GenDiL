// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

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
 * @param[in,out] hessenberg    Pre‐allocated “flat” array of length (restart+1)*restart,
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
    typename Vector
>
GENDIL_HOST_DEVICE
auto GMRES_no_alloc(
    const Operator &         A,
    const RHSType &           rhs,
    const DotType &          dot,
    const Integer            max_iters,
    const Integer            restart,
    const Real               tol,
    Vector &                 x,
    Vector *                 V,           // array of (restart+1) Vectors
    Vector &                 w,           // single workspace Vector
    Real *                   hessenberg,  // length = (restart+1)*restart
    Real *                   cs,          // length = restart
    Real *                   sn,          // length = restart
    Real *                   e1_rhs,      // length = restart+1
    Real *                   y            // length = restart
)
{
    using Result = std::tuple<bool, Integer, Real>;
    const Real    tol_rel = tol;

    // 1) Compute initial residual r0 = rhs - A*x; store in V[0]
    A(x, w);
    V[0] = rhs;
    V[0] -= w; 
    Real beta = Sqrt(dot(V[0], V[0]));
    if (beta == 0.0)
    {
        // Exact solution already
        return Result{ true, 0, 0.0 };
    }

    // 2) Compute ‖rhs‖ for relative tol
    Real norm_rhs = Sqrt(dot(rhs, rhs));
    if (norm_rhs == 0.0)
    {
        x = 0.0;
        return Result{ true, 0, 0.0 };
    }

    Integer total_iters = 0;

    // Outer restart loop
    while (total_iters < max_iters)
    {
        for (Integer i = 0; i < restart; ++i)
        {
            cs[i] = 1.0;
            sn[i] = 0.0;
        }

        // Normalize v0 = r0 / beta
        const Real inv_beta = 1.0 / beta;
        V[0] *= inv_beta;

        // Initialize small‐RHS for least‐squares: e1_rhs = [beta, 0, 0, ..., 0]^T
        for (Integer i = 0; i <= restart; ++i) e1_rhs[i] = 0.0;
        e1_rhs[0] = beta;

        Integer m = 0;  // actual Arnoldi dimension

        // Inner Arnoldi loop (at most 'restart' steps, and ≤ max_iters total)
        for ( ; m < restart && total_iters < max_iters; ++m, ++total_iters)
        {
            // 3) w = A * v_m
            A(V[m], w);

            // 4) Modified Gram–Schmidt: orthonormalize w against V[0..m]
            for (Integer i = 0; i <= m; ++i)
            {
                Real hij = dot(V[i], w);
                hessenberg[m*(restart+1) + i] = hij;
                Axpy(-hij, V[i], w);
            }

            Real h_next = Sqrt(dot(w, w));
            hessenberg[m*(restart+1) + (m+1)] = h_next;

            // Check for happy breakdown (w ≈ 0)
            if (h_next <= std::numeric_limits<Real>::epsilon() * norm_rhs)
            {
                // We found an invariant subspace; keep m at this value,
                // and do NOT increment m further.  Break out.
                break;
            }

            // 5) Normalize v_{m+1} = w / h_next
            V[m+1] = w;
            V[m+1] *= (1.0 / h_next );

            // 6) Apply all previous Givens rotations to H[0..m+1, m]
            for (Integer i = 0; i < m; ++i)
            {
                Real temp1 = cs[i] * hessenberg[m*(restart+1) + i]
                           + sn[i] * hessenberg[m*(restart+1) + (i+1)];
                Real temp2 = -sn[i] * hessenberg[m*(restart+1) + i]
                            + cs[i] * hessenberg[m*(restart+1) + (i+1)];
                hessenberg[m*(restart+1) + i]     = temp1;
                hessenberg[m*(restart+1) + (i+1)] = temp2;
            }

            // 7) Generate new Givens to zero H(m+1,m)
            {
                Real h_mm   = hessenberg[m*(restart+1) + m];
                Real h_m1m  = hessenberg[m*(restart+1) + (m+1)];
                Real rho    = Sqrt(h_mm*h_mm + h_m1m*h_m1m);
                cs[m] =  h_mm / rho;
                sn[m] =  h_m1m / rho;

                // Rotate into H(m,m) and H(m+1,m)
                hessenberg[m*(restart+1) + m]     = cs[m] * h_mm + sn[m] * h_m1m;
                hessenberg[m*(restart+1) + (m+1)] = 0.0;

                // Also rotate the small RHS entries e1_rhs[m], e1_rhs[m+1]
                Real rho_m   = e1_rhs[m];
                Real rho_m1  = e1_rhs[m+1];
                e1_rhs[m]   =  cs[m] * rho_m   + sn[m] * rho_m1;
                e1_rhs[m+1] = -sn[m] * rho_m   + cs[m] * rho_m1;
            }

            // 8) Check convergence:  |e1_rhs[m+1]| / norm_rhs < tol_rel?
            Real rel_res = Abs(e1_rhs[m+1]) / norm_rhs;
            if (rel_res <= tol_rel)
            {
                // We have converged in m+1 steps.  Keep 'm' as is and exit.
                break;
            }
        }

        // At this point, 'm' is the number of Arnoldi steps taken.
        // (It ranges from 0..restart.) 
        // We now solve the small upper‐Hessenberg least‐squares of size (m+1)×m
        // (or smaller if we broke early).

        // Back‐substitution on the m×m upper‐triangular leading block of H:
        // R(i,i)*y[i] + Σ_{j=i+1..(m-1)} R(i,j)*y[j] = e1_rhs[i],  for i=m-1..0
        if (m > 0)
        {
            for (Integer i = m;  i-- > 0; )
            {
                Real sum = e1_rhs[i];
                for (Integer j = i+1;  j < m;  ++j)
                {
                    sum -= hessenberg[j*(restart+1) + i] * y[j];
                }
                y[i] = sum / hessenberg[i*(restart+1) + i];
            }
        }
        else // m == 0
        {
            // Because the first Arnoldi step had h_next = 0, A·v0 = beta·v0.
            // That means v0 is already an eigenvector.  We “update” x once by adding
            // y[0] = e1_rhs[0]/hessenberg[0*(restart+1)+0] = beta / (h_{00}).
            // But h_{00} = dot(A·v0, v0) = (v0^T A v0) = beta, so y[0] = 1.0.
            // Hence x += V[0] * 1.0 is the exact (one‐step) correction and we are done.
            Axpy( e1_rhs[0] / hessenberg[0*(restart+1) + 0], V[0], x );
            Real final_rel_res = 0.0;  // exact convergence
            return Result{ true, total_iters, final_rel_res };
        }

        // 9) Update solution:  x += Σ_{i=0..(m-1)} y[i] * V[i]
        for (Integer i = 0; i < m; ++i)
        {
            Axpy(y[i], V[i], x);
        }

        // 10) Compute final relative residual = |e1_rhs[m]| / norm_rhs
        Real final_rel_res = Abs(e1_rhs[m]) / norm_rhs;
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

        // Compute new r0 → store in V[0]
        A(x, w);
        V[0] = rhs;
        V[0] -= w;
        beta = Sqrt(dot(V[0], V[0]));
        if (beta == 0.0)
        {
            // Exact solution reached after an outer cycle
            return Result{ true, total_iters, 0.0 };
        }

        // Loop again, re‐normalizing V[0] = r0 / beta, resetting e1_rhs, etc.
        // std::cout << "Iter: " << total_iters << ", residual = " << final_rel_res <<  std::endl;
    }

    // If we reach here, max_iters was exhausted.  Recompute actual residual:
    A(x, w);
    V[0] = rhs;
    V[0] -= w;
    Real rnorm = Sqrt(dot(V[0], V[0]));
    return Result{ false, total_iters, rnorm / Sqrt(dot(rhs, rhs)) };
}

}
