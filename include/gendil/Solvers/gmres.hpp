// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <limits>
#include "gendil/prelude.hpp"
#include "gendil/Solvers/iterativesolverresult.hpp"

namespace gendil {

/**
 * @brief Solve A x = rhs using restarted GMRES without any internal allocations.
 *
 * This version requires the caller to hand in *all* needed scratch space.
 * No `resize`, `new`, or other dynamic allocation happens inside.
 *
 * The implementation uses:
 * - Arnoldi with modified Gram-Schmidt,
 * - incremental Givens rotations for the small QR factorization,
 * - a delayed solution update at the end of each restart cycle.
 *
 * The function returns the *true* residual norm recomputed from `rhs - A*x`
 * after each completed GMRES cycle.
 *
 * Special case:
 * - If `||rhs|| == 0`, the tolerance is interpreted as an **absolute**
 *   residual tolerance on `||rhs - A*x||`.
 *
 * @tparam Operator   Functor for matrix-vector product: w = A * v
 * @tparam RHSType    RHS vector-like type
 * @tparam DotType    Dot-product functor: returns Real dot(const Vector&, const Vector&)
 * @tparam VectorArray Array-like container holding `(restart+1)` work vectors
 * @tparam Vector     Mutable vector type (e.g. std::vector<Real> or a GPU vector)
 * @tparam Array      Array-like type used for the small dense work arrays
 *
 * @param[in]     A            Linear operator: A(v, w) ⟹ w = A·v
 * @param[in]     rhs          Right-hand-side vector
 * @param[in]     dot          Dot-product function (performs any global reductions)
 * @param[in]     max_iters    Maximum total GMRES iterations (counting restarts)
 * @param[in]     restart      GMRES restart length (m). Typically m ≪ dimension.
 * @param[in]     tol          Relative residual tolerance, except if `||rhs|| == 0`
 *                             in which case it is interpreted as an absolute tolerance
 *
 * @param[in,out] x            On entry: initial guess. On exit: approximate solution.
 *
 * @param[in,out] V            Pre-allocated array of `(restart+1)` Vectors,
 *                             each of size = x.size(). Used to store the Krylov basis.
 * @param[in,out] w            Pre-allocated workspace Vector (size = x.size()),
 *                             used for computing A·v_j and residuals.
 * @param[in,out] hessenberg   Pre-allocated flat array of length `(restart+1)*restart`,
 *                             storing H in column-major order; i.e.
 *                             H(i,j) = hessenberg[j*(restart+1) + i].
 * @param[in,out] cs           Pre-allocated array of length `restart` for Givens cosines.
 * @param[in,out] sn           Pre-allocated array of length `restart` for Givens sines.
 * @param[in,out] e1_rhs       Pre-allocated array of length `(restart+1)` for the small QR RHS.
 * @param[in,out] y            Pre-allocated array of length `restart` for the reduced unknowns.
 *
 * @return IterativeSolverResult
 *         - converged : true if the residual norm satisfies the requested tolerance
 *         - num_iters : total number of GMRES iterations performed
 *         - residual  : final true residual norm, relative to `||rhs||` when `||rhs|| > 0`,
 *                       otherwise absolute
 */
template<
   typename Operator,
   typename RHSType,
   typename DotType,
   typename VectorArray,
   typename Vector,
   typename Array
>
auto GMRES_no_alloc(
   const Operator & A,
   const RHSType &  rhs,
   const DotType &  dot,
   const Integer    max_iters,
   const Integer    restart,
   const Real       tol,
   Vector &         x,
   VectorArray &    V,           // array of (restart+1) Vectors
   Vector &         w,           // single workspace Vector
   Array &          hessenberg,  // length = (restart+1)*restart
   Array &          cs,          // length = restart
   Array &          sn,          // length = restart
   Array &          e1_rhs,      // length = restart+1
   Array &          y            // length = restart
)
{
   const Real eps = std::numeric_limits<Real>::epsilon();

   auto H = [&](const Integer i, const Integer j) -> decltype(auto)
   {
      return hessenberg[j * (restart + 1) + i];
   };

   auto ComputeResidualInV0 = [&]() -> Real
   {
      A(x, w);
      V[0] = rhs;
      V[0] -= w; // V[0] = rhs - A*x
      return Sqrt(dot(V[0], V[0]));
   };

   // Degenerate iteration budgets
   if (max_iters <= 0 || restart <= 0)
   {
      const Real norm_rhs = Sqrt(dot(rhs, rhs));
      const Real denom    = (norm_rhs > Real(0)) ? norm_rhs : Real(1);
      const Real rnorm    = ComputeResidualInV0();
      return IterativeSolverResult{ rnorm / denom <= tol, 0, rnorm / denom };
   }

   // Residual scaling:
   // - relative if ||rhs|| > 0
   // - absolute if ||rhs|| == 0
   const Real norm_rhs = Sqrt(dot(rhs, rhs));
   const Real denom    = (norm_rhs > Real(0)) ? norm_rhs : Real(1);

   // Initial residual r0 = rhs - A*x, stored in V[0]
   Real beta = ComputeResidualInV0();
   Real rel_res = beta / denom;

   if (rel_res <= tol)
   {
      return IterativeSolverResult{ true, 0, rel_res };
   }

   Integer total_iters = 0;

   while (total_iters < max_iters)
   {
      // Normalize v0 = r / ||r||
      V[0] *= (Real(1) / beta);

      // Reset Givens data and reduced RHS
      for (Integer i = 0; i < restart; ++i)
      {
         cs[i] = Real(1);
         sn[i] = Real(0);
         y[i]  = Real(0);
      }
      for (Integer i = 0; i <= restart; ++i)
      {
         e1_rhs[i] = Real(0);
      }
      e1_rhs[0] = beta;

      Integer k = 0; // actual number of usable Krylov vectors in this cycle

      for (Integer m = 0; m < restart && total_iters < max_iters; ++m)
      {
         ++total_iters;

         // 1) w = A * v_m
         A(V[m], w);

         // 2) Arnoldi orthogonalization (modified Gram-Schmidt)
         for (Integer i = 0; i <= m; ++i)
         {
               const Real hij = dot(V[i], w);
               H(i, m) = hij;
               Axpy(-hij, V[i], w);
         }

         const Real h_next = Sqrt(dot(w, w));
         H(m + 1, m) = h_next;

         // 3) Apply all previous Givens rotations to current column
         for (Integer i = 0; i < m; ++i)
         {
               const Real h_im   = H(i,     m);
               const Real h_ip1m = H(i + 1, m);

               H(i,     m) =  cs[i] * h_im + sn[i] * h_ip1m;
               H(i + 1, m) = -sn[i] * h_im + cs[i] * h_ip1m;
         }

         // 4) Generate new Givens rotation for entries H(m,m), H(m+1,m)
         const Real h_mm  = H(m,     m);
         const Real h_m1m = H(m + 1, m);
         const Real rho   = Sqrt(h_mm * h_mm + h_m1m * h_m1m);

         // "Happy breakdown" or exact linear dependence of the new column.
         // If rho == 0 after previous rotations, the current column contributes
         // no new search direction. In that case, the best approximation in this
         // cycle is the one from the previous Krylov dimension k = m.
         if (rho <= eps * beta)
         {
               H(m,     m) = Real(0);
               H(m + 1, m) = Real(0);
               k = m;
               break;
         }

         cs[m] = h_mm  / rho;
         sn[m] = h_m1m / rho;

         H(m,     m) = rho;
         H(m + 1, m) = Real(0);

         // Apply same rotation to the reduced RHS
         const Real g_m  = e1_rhs[m];
         const Real g_m1 = e1_rhs[m + 1];

         e1_rhs[m]     =  cs[m] * g_m + sn[m] * g_m1;
         e1_rhs[m + 1] = -sn[m] * g_m + cs[m] * g_m1;

         k = m + 1;

         // Cheap residual estimate from the QR-updated reduced problem
         const Real rel_est = Abs(e1_rhs[m + 1]) / denom;

         // If the new Arnoldi vector is numerically zero, the Krylov subspace is
         // invariant and this cycle is complete. We break *after* incorporating
         // the current column if rho was nonzero.
         const bool happy_breakdown = (h_next <= eps * beta);

         if (happy_breakdown || rel_est <= tol)
         {
               break;
         }

         // Continue only if another basis vector is needed
         V[m + 1] = w;
         V[m + 1] *= (Real(1) / h_next);
      }

      // No usable search direction was produced in this cycle.
      // Then the iterate cannot be improved by this GMRES restart.
      if (k == 0)
      {
         const Real rnorm = ComputeResidualInV0();
         return IterativeSolverResult{ rnorm / denom <= tol, total_iters, rnorm / denom };
      }

      // Back-substitution in the k x k upper-triangular reduced system.
      //
      // After the logic above, the leading k x k block should be nonsingular in
      // the standard GMRES case. We still protect against tiny diagonals to avoid
      // invalid floating-point operations in nearly singular situations.
      for (Integer i = k; i-- > 0; )
      {
         Real sum = e1_rhs[i];
         for (Integer j = i + 1; j < k; ++j)
         {
               sum -= H(i, j) * y[j];
         }

         const Real diag = H(i, i);
         if (Abs(diag) <= eps * beta)
         {
               y[i] = Real(0);
         }
         else
         {
               y[i] = sum / diag;
         }
      }

      // Update x = x + V(:,0:k-1) * y
      for (Integer i = 0; i < k; ++i)
      {
         Axpy(y[i], V[i], x);
      }

      // Recompute the *true* residual for the returned result and for restart logic
      beta    = ComputeResidualInV0();
      rel_res = beta / denom;

      if (rel_res <= tol)
      {
         return IterativeSolverResult{ true, total_iters, rel_res };
      }

      if (total_iters >= max_iters)
      {
         return IterativeSolverResult{ false, total_iters, rel_res };
      }

      // Otherwise continue to next restart cycle with V[0] holding the new residual
   }

   // Defensive fallback
   const Real rnorm = ComputeResidualInV0();
   return IterativeSolverResult{ rnorm / denom <= tol, total_iters, rnorm / denom };
}

} // namespace gendil