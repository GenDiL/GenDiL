// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <utility>

#include "gendil/Solvers/iterativesolverresult.hpp"

/**
 * @file conjugate_gradient_diagnostic.hpp
 * @brief Diagnostic version of the Conjugate Gradient (CG) solver with comprehensive iteration tracking
 *
 * # Overview
 *
 * This file provides a diagnostic implementation of the Conjugate Gradient method that captures
 * detailed information about CG iteration behavior. Unlike the standard CG solver (which focuses
 * solely on finding the solution), this diagnostic version tracks:
 *
 * - **Numerical stability**: Recursive vs. true residuals, rounding error accumulation
 * - **Convergence analysis**: Iteration-by-iteration residual norms, stagnation detection
 * - **Spectral properties**: Ritz values, condition number estimates from the Lanczos connection
 * - **Orthogonality loss**: How numerical errors degrade the Krylov basis over iterations
 *
 * # Why Use This?
 *
 * The diagnostic CG is valuable for:
 *
 * 1. **Understanding convergence behavior**: Visualize how CG converges (or stagnates) for your problem
 * 2. **Debugging numerical issues**: Identify when rounding errors become significant
 * 3. **Comparing precisions**: See how different floating-point types (float, double, bfloat16) affect convergence
 * 4. **Analyzing matrix properties**: Estimate condition numbers and eigenvalue distributions
 * 5. **Research and education**: Study CG behavior in detail for publications or teaching
 *
 * # Key Concepts
 *
 * ## Recursive vs. True Residuals
 *
 * - **Recursive residual**: Computed using the CG recurrence r_{k+1} = r_k - alpha * A * p_k
 *   - Fast to compute (no extra matrix-vector product)
 *   - Accumulates rounding errors over iterations
 *
 * - **True residual**: Recomputed from scratch as r = b - A*x
 *   - Requires an extra matrix-vector product per iteration
 *   - Shows the actual residual (free from accumulated rounding errors)
 *
 * - **Residual gap**: Difference between recursive and true residuals
 *   - Measures accumulated rounding error
 *   - Large gap indicates numerical instability
 *
 * ## Lanczos Connection
 *
 * CG and the Lanczos method are mathematically equivalent. From CG's alpha and beta coefficients,
 * we can recover the Lanczos tridiagonal matrix T_k, whose eigenvalues (Ritz values) approximate
 * the eigenvalues of the original matrix A.
 *
 * ## Ritz Values
 *
 * Eigenvalues of the Lanczos tridiagonal T_k that approximate A's eigenvalues. From these we estimate:
 * - Condition number: max_ritz / min_ritz
 * - Convergence rate: Related to eigenvalue distribution
 * - Spectral gaps: Separation between eigenvalues
 *
 * # Usage Example
 *
 * ```cpp
 * // Set up your linear system A*x = b
 * Vector x(n), tmp(n), z(n), residual(n), p(n);
 * x = 0.0; tmp = 0.0; z = 0.0; residual = 0.0; p = 0.0;
 *
 * // Configure diagnostics
 * CGDiagnosticOptions opts;
 * opts.recompute_true_residual_each_iter = true;  // Track residual gap
 * opts.store_basis = false;                        // Save memory (disable orthogonality checks)
 * opts.estimate_ritz = true;                       // Compute Ritz values
 * opts.store_snapshots = false;                    // Don't store full vectors
 *
 * // Solve with diagnostics
 * auto diag = ConjugateGradientDiagnose(A, b, dot, max_iters, tol, x, tmp, z, residual, p, opts);
 *
 * // Analyze results
 * std::cout << "Converged: " << diag.success << "\n";
 * std::cout << "Iterations: " << diag.iterations << "\n";
 * std::cout << "Condition number estimate: "
 *           << diag.steps.back().estimated_condition << "\n";
 *
 * // Plot convergence
 * for (const auto& step : diag.steps) {
 *     std::cout << step.iter << " "
 *               << step.relative_recursive_residual << " "
 *               << step.relative_true_residual << "\n";
 * }
 * ```
 *
 * # Performance Considerations
 *
 * - Setting `recompute_true_residual_each_iter = true` doubles matrix-vector products per iteration
 * - Setting `store_basis = true` requires O(iterations * vector_size) memory
 * - Setting `estimate_ritz = true` adds O(iterations^2) work per iteration (small for reasonable iteration counts)
 * - Setting `store_snapshots = true` stores full solution/residual vectors at each iteration
 *
 * For production solves, use the standard `ConjugateGradient()` instead. Use this diagnostic version
 * for analysis, debugging, and understanding convergence behavior.
 *
 * # Mathematical Background
 *
 * References:
 * - Paige & Saunders (1975): "Solution of sparse indefinite systems of linear equations"
 * - Greenbaum (1997): "Iterative Methods for Solving Linear Systems"
 * - Meurant & Duintjer Tebbens (2020): "The role of the Lanczos tridiagonal matrix in the CG algorithm"
 *
 * Prerequisites:
 * - Real, Integer types defined by GenDiL
 * - Axpy(alpha, x, y) function: y += alpha * x
 * - Vector type supporting: copy, +=, -=, *= scalar, assignment
 * - dot(u, v) function returning Real
 * - Operator A such that A(in, out) computes out = A * in
 */

namespace gendil {

// ------------------------------------------------------------
// Options
// ------------------------------------------------------------

/**
 * @brief Configuration options for diagnostic CG
 *
 * Controls which diagnostics to collect during CG iterations. Each option
 * has different memory and performance trade-offs. Enable only what you need
 * for your analysis.
 *
 * For production solves where you only need the solution, use the standard
 * ConjugateGradient() function instead.
 */
struct CGDiagnosticOptions
{
   /**
    * @brief Recompute true residual r = b - A*x at each iteration
    *
    * When enabled, computes the true residual from scratch at each iteration
    * by explicitly computing r = b - A*x, in addition to the recursive residual
    * that CG maintains via the update r_{k+1} = r_k - alpha * A * p_k.
    *
    * **Cost**: Doubles the number of matrix-vector products per iteration
    *            (one for the CG update, one for the true residual)
    *
    * **Use when**: You want to track rounding error accumulation (residual gap),
    *               debug convergence stagnation, or compare different precisions
    *
    * **Default**: true
    */
   bool recompute_true_residual_each_iter = true;

   /**
    * @brief Store normalized Krylov basis vectors for orthogonality checks
    *
    * When enabled, stores normalized residuals, search directions p_k, and
    * A*p_k vectors at each iteration to compute loss of orthogonality metrics.
    *
    * **Cost**: O(iterations * vector_size) memory. For n=10^6 and 100 iterations,
    *           stores ~2.4GB (assuming double precision)
    *
    * **Use when**: You want to measure numerical degradation of the Krylov basis,
    *               typically for research or deep numerical analysis
    *
    * **Default**: true (but consider disabling for large problems or many iterations)
    */
   bool store_basis = true;

   /**
    * @brief Build Lanczos tridiagonal T_k and compute Ritz values
    *
    * When enabled, recovers the Lanczos tridiagonal matrix from CG coefficients
    * and computes its eigenvalues (Ritz values) to estimate condition number and
    * eigenvalue distribution.
    *
    * **Cost**: O(iterations^2) per iteration for eigenvalue computation
    *           (negligible for typical iteration counts < 1000)
    *
    * **Use when**: You want condition number estimates, eigenvalue approximations,
    *               or convergence rate analysis
    *
    * **Default**: true
    */
   bool estimate_ritz = true;

   /**
    * @brief Store complete solution/residual/search snapshots at each iteration
    *
    * When enabled, stores full copies of x_k, r_k, and p_k at each iteration.
    * Useful for detailed post-processing or visualization.
    *
    * **Cost**: O(iterations * vector_size * 3) memory. For n=10^6 and 100 iterations,
    *           stores ~2.4GB (assuming double precision)
    *
    * **Use when**: You need to analyze the solution trajectory in detail or create
    *               animations of convergence
    *
    * **Default**: false (very memory-intensive, rarely needed)
    */
   bool store_snapshots = false;
};

// ------------------------------------------------------------
// Per-iteration diagnostics
// ------------------------------------------------------------

/**
 * @brief Diagnostic data captured at a single CG iteration
 *
 * Contains comprehensive diagnostic information about the CG iteration state,
 * including convergence metrics, spectral estimates, and numerical stability
 * indicators. Fields are organized into logical groups for clarity.
 *
 * This structure captures the state *after* iteration `iter` has completed.
 */
template <typename Vector>
struct CGIterationDiagnostics
{
   Integer iter = 0; ///< Iteration number (state after this iteration completes)

   // ---- Basic CG Recurrence Coefficients ----
   // These are the fundamental scalars that define the CG update

   Real alpha = 0.0;   ///< Step size along search direction: x_{k+1} = x_k + alpha * p_k
   Real beta  = 0.0;   ///< Conjugation coefficient: p_{k+1} = r_{k+1} + beta * p_k
   Real pAp   = 0.0;   ///< p_k^T A p_k (denominator in alpha computation)
   Real rr_old = 0.0;  ///< r_k^T r_k (residual inner product before update)
   Real rr_new = 0.0;  ///< r_{k+1}^T r_{k+1} (residual inner product after update)

   // ---- Residual Norms ----
   // Different ways to measure convergence and rounding error

   Real recursive_residual_norm = 0.0;      ///< ||r_k|| computed via CG recurrence
   Real relative_recursive_residual = 0.0;  ///< ||r_k|| / ||b|| (recursive)
   Real true_residual_norm = 0.0;           ///< ||b - A*x|| computed from scratch
   Real relative_true_residual = 0.0;       ///< ||b - A*x|| / ||b|| (true)
   Real residual_gap_norm = 0.0;            ///< ||r_recursive - r_true|| (rounding error)

   // ---- Vector Norms ----
   // Diagnostic norms for solution and search directions

   Real x_norm  = 0.0;  ///< ||x_k|| (solution norm)
   Real p_norm  = 0.0;  ///< ||p_k|| (search direction norm)
   Real Ap_norm = 0.0;  ///< ||A*p_k|| (operator applied to search direction)

   // ---- Quadratic Quantities ----
   // Energy functional components: minimizing 0.5 * x^T A x - b^T x

   Real xAx = 0.0;        ///< x_k^T A x_k (quadratic term)
   Real bTx = 0.0;        ///< b^T x_k (linear term)
   Real objective = 0.0;  ///< 0.5 * x^T A x - b^T x (CG objective function value)

   // ---- Loss of Orthogonality / Conjugacy ----
   // Measures numerical degradation of the Krylov basis
   // Values near 0 indicate good numerical behavior; values near 1 indicate severe loss

   Real max_residual_orthogonality = 0.0;  ///< max |<r_k, r_j>| / (||r_k|| ||r_j||) over j<k
   Real max_search_A_conjugacy = 0.0;      ///< max |<p_k, A*p_j>| / sqrt(<p_k,A*p_k><p_j,A*p_j>) over j<k

   // ---- Lanczos Tridiagonal Matrix ----
   // Recovered from CG coefficients; equivalent representation of CG iteration

   std::vector<Real> lanczos_diag;      ///< Diagonal of T_k
   std::vector<Real> lanczos_offdiag;   ///< Off-diagonal of T_k
   Real lanczos_trailing = 0.0;         ///< Coupling to next Lanczos vector (for residual estimates)

   // ---- Ritz Values and Estimates ----
   // Eigenvalue approximations and condition number estimates from T_k

   std::vector<Real> ritz_values;              ///< Eigenvalues of T_k (approximate eigenvalues of A)
   std::vector<Real> ritz_residual_estimates;  ///< ||A*z_i - theta_i*z_i|| estimates for each Ritz pair
   std::vector<Real> ritz_gap_estimates;       ///< Distance to nearest neighbor Ritz value
   Real min_ritz = 0.0;                        ///< Smallest Ritz value (approximates lambda_min(A))
   Real max_ritz = 0.0;                        ///< Largest Ritz value (approximates lambda_max(A))
   Real estimated_condition = 0.0;             ///< max_ritz / min_ritz (condition number estimate)

   // ---- Optional Snapshots ----
   // Full vector storage at this iteration (only stored if store_snapshots=true)

   Vector x_snapshot;        ///< Solution vector x_k
   Vector residual_snapshot; ///< Residual vector r_k
   Vector p_snapshot;        ///< Search direction p_k
};

// ------------------------------------------------------------
// Global diagnostics
// ------------------------------------------------------------

/**
 * @brief Complete diagnostic results for a CG solve
 *
 * Inherits standard iterative solver results (success, iterations, relative_error)
 * from IterativeSolverResult and adds CG-specific global diagnostics plus detailed
 * per-iteration data.
 *
 * The `steps` vector contains CGIterationDiagnostics for each iteration, allowing
 * comprehensive post-solve analysis of:
 * - Convergence behavior (residual norms over iterations)
 * - Numerical stability (residual gap, orthogonality loss)
 * - Spectral properties (Ritz values, condition number estimates)
 *
 * **Usage Pattern:**
 * ```cpp
 * auto diag = ConjugateGradientDiagnose(...);
 *
 * // Check convergence
 * if (diag.success) {
 *     std::cout << "Converged in " << diag.iterations << " iterations\n";
 * }
 *
 * // Analyze convergence history
 * for (const auto& step : diag.steps) {
 *     std::cout << step.iter << " " << step.relative_true_residual << "\n";
 * }
 *
 * // Check condition number
 * std::cout << "Estimated condition: " << diag.steps.back().estimated_condition << "\n";
 * ```
 */
template <typename Vector>
struct CGDiagnostics : public IterativeSolverResult
{
   // Inherits from IterativeSolverResult:
   //   bool success          - true if CG converged within tolerance
   //   Integer iterations    - number of iterations performed
   //   Real relative_error   - final relative residual norm

   Real rhs_norm = 0.0;                          ///< Norm of right-hand side ||b||
   Real initial_recursive_residual_norm = 0.0;   ///< Initial ||r_0|| (recursive)
   Real initial_relative_recursive_residual = 0.0; ///< Initial ||r_0|| / ||b|| (recursive)
   Real initial_true_residual_norm = 0.0;        ///< Initial ||b - A*x_0|| (true)
   Real initial_relative_true_residual = 0.0;    ///< Initial ||b - A*x_0|| / ||b|| (true)

   /**
    * @brief Per-iteration diagnostic data
    *
    * Vector of CGIterationDiagnostics, one entry per iteration.
    * steps[k] contains diagnostics after iteration k has completed.
    *
    * Empty if CG converged in 0 iterations (initial guess already satisfied tolerance).
    */
   std::vector< CGIterationDiagnostics<Vector> > steps;
};

// ------------------------------------------------------------
// Small dense symmetric eigensolver (Jacobi) for Ritz analysis
// ------------------------------------------------------------

/**
 * @brief Result of symmetric eigenvalue decomposition
 *
 * Eigenvalues are sorted in ascending order.
 * eigenvectors[i] corresponds to eigenvalues[i].
 * Each eigenvector is stored as a column vector.
 */
struct SymmetricEigenResult
{
   std::vector<Real> eigenvalues;  ///< Eigenvalues in ascending order
   std::vector< std::vector<Real> > eigenvectors;  ///< eigenvectors[i] = i-th eigenvector (column)
};

/**
 * @brief Compute eigendecomposition of small dense symmetric matrix using Jacobi method
 *
 * Classical Jacobi rotation method for symmetric eigenvalue problems. Suitable for
 * small matrices (n < 100). Used internally to compute Ritz values from the Lanczos
 * tridiagonal matrix.
 *
 * The algorithm repeatedly applies Jacobi rotations to zero out the largest off-diagonal
 * element until all off-diagonal elements are below tolerance.
 *
 * @param A Symmetric matrix (will be modified during computation)
 * @param max_sweeps Maximum number of Jacobi sweeps (default: 100)
 * @param tol Convergence tolerance for off-diagonal elements (default: 1e-30)
 * @return SymmetricEigenResult with eigenvalues (ascending order) and eigenvectors
 */
inline
auto SymmetricEigenDecompositionJacobi(
   std::vector< std::vector<Real> > A,
   const Integer max_sweeps = 100,
   const Real tol = Real(1e-30) )
   -> SymmetricEigenResult
{
   const Integer n = static_cast<Integer>(A.size());

   std::vector< std::vector<Real> > Q(n, std::vector<Real>(n, Real(0)));
   for (Integer i = 0; i < n; ++i) {
      Q[i][i] = Real(1);
   }

   if (n == 0) {
      return {};
   }

   for (Integer sweep = 0; sweep < max_sweeps; ++sweep)
   {
      Integer p = 0, q = 1;
      Real max_offdiag = Real(0);

      for (Integer i = 0; i < n; ++i) {
         for (Integer j = i + 1; j < n; ++j) {
            const Real aij = std::abs(A[i][j]);
            if (aij > max_offdiag) {
               max_offdiag = aij;
               p = i;
               q = j;
            }
         }
      }

      if (n == 1 || max_offdiag < tol) {
         break;
      }

      const Real app = A[p][p];
      const Real aqq = A[q][q];
      const Real apq = A[p][q];

      const Real tau = (aqq - app) / (Real(2) * apq);
      const Real t =
         (tau >= Real(0))
            ? Real(1) / (tau + std::sqrt(Real(1) + tau*tau))
            : Real(-1) / (-tau + std::sqrt(Real(1) + tau*tau));

      const Real c = Real(1) / std::sqrt(Real(1) + t*t);
      const Real s = t * c;

      // Update A
      for (Integer k = 0; k < n; ++k)
      {
         if (k == p || k == q) continue;

         const Real akp = A[k][p];
         const Real akq = A[k][q];

         A[k][p] = c * akp - s * akq;
         A[p][k] = A[k][p];

         A[k][q] = s * akp + c * akq;
         A[q][k] = A[k][q];
      }

      const Real new_app = c*c*app - Real(2)*s*c*apq + s*s*aqq;
      const Real new_aqq = s*s*app + Real(2)*s*c*apq + c*c*aqq;

      A[p][p] = new_app;
      A[q][q] = new_aqq;
      A[p][q] = Real(0);
      A[q][p] = Real(0);

      // Update eigenvector matrix Q
      for (Integer k = 0; k < n; ++k)
      {
         const Real qkp = Q[k][p];
         const Real qkq = Q[k][q];
         Q[k][p] = c * qkp - s * qkq;
         Q[k][q] = s * qkp + c * qkq;
      }
   }

   SymmetricEigenResult result;
   result.eigenvalues.resize(n);
   result.eigenvectors.resize(n, std::vector<Real>(n, Real(0)));

   for (Integer i = 0; i < n; ++i) {
      result.eigenvalues[i] = A[i][i];
      for (Integer k = 0; k < n; ++k) {
         result.eigenvectors[i][k] = Q[k][i]; // column i
      }
   }

   // sort ascending
   std::vector<Integer> perm(n);
   for (Integer i = 0; i < n; ++i) perm[i] = i;

   std::sort(perm.begin(), perm.end(),
      [&](Integer a, Integer b) {
         return result.eigenvalues[a] < result.eigenvalues[b];
      });

   SymmetricEigenResult sorted;
   sorted.eigenvalues.resize(n);
   sorted.eigenvectors.resize(n);

   for (Integer i = 0; i < n; ++i) {
      sorted.eigenvalues[i] = result.eigenvalues[perm[i]];
      sorted.eigenvectors[i] = result.eigenvectors[perm[i]];
   }

   return sorted;
}

// ------------------------------------------------------------
// CG -> Lanczos tridiagonal recovery
// ------------------------------------------------------------

/**
 * @brief Lanczos tridiagonal matrix T_k recovered from CG coefficients
 *
 * The Conjugate Gradient method and the Lanczos method are mathematically
 * equivalent. From CG's alpha and beta coefficients, we can recover the
 * Lanczos tridiagonal matrix T_k whose eigenvalues (Ritz values) approximate
 * the eigenvalues of the original matrix A.
 *
 * **Mathematical relationship for unpreconditioned CG:**
 * - diag[0]   = 1/alpha_0
 * - diag[i]   = 1/alpha_i + beta_{i-1}/alpha_{i-1}, for i >= 1
 * - offdiag[i]= sqrt(beta_i)/alpha_i, for i = 0..m-2
 * - trailing  = sqrt(beta_{m-1})/alpha_{m-1}
 *
 * **Structure:**
 * T_k is a symmetric tridiagonal matrix:
 * - diag contains the diagonal elements
 * - offdiag contains the super/sub-diagonal elements (symmetric)
 * - trailing stores the coupling to the next Lanczos vector (used for residual estimates)
 *
 * The eigenvalues of T_k (Ritz values) provide increasingly accurate approximations
 * to A's eigenvalues as k increases.
 */
struct LanczosTridiagonalSnapshot
{
   std::vector<Real> diag;      ///< Diagonal elements of T_k
   std::vector<Real> offdiag;   ///< Off-diagonal elements of T_k (length = diag.size()-1)
   Real trailing = 0.0;         ///< Coupling to next Lanczos vector (for Ritz residual estimates)
};

/**
 * @brief Recover Lanczos tridiagonal matrix from CG coefficients
 *
 * Exploits the mathematical equivalence between CG and the Lanczos method to
 * reconstruct the Lanczos tridiagonal matrix T_k from CG's alpha and beta
 * coefficients. The eigenvalues of T_k (Ritz values) approximate A's eigenvalues.
 *
 * **Mathematical relationship for unpreconditioned CG:**
 * - diag[0]   = 1/alpha_0
 * - diag[i]   = 1/alpha_i + beta_{i-1}/alpha_{i-1}, for i >= 1
 * - offdiag[i]= sqrt(beta_i)/alpha_i, for i = 0..m-2
 * - trailing  = sqrt(beta_{m-1})/alpha_{m-1} (coupling to next Lanczos vector)
 *
 * @param alpha_hist History of CG alpha coefficients (step sizes)
 * @param beta_hist History of CG beta coefficients (conjugation parameters)
 * @param m Number of iterations to reconstruct (typically alpha_hist.size())
 * @return LanczosTridiagonalSnapshot containing T_k
 */
inline
auto BuildLanczosFromCGCoefficients(
   const std::vector<Real> & alpha_hist,
   const std::vector<Real> & beta_hist,
   const Integer m )
   -> LanczosTridiagonalSnapshot
{
   LanczosTridiagonalSnapshot out;

   if (m <= 0) {
      return out;
   }

   out.diag.resize(m, Real(0));
   if (m > 1) {
      out.offdiag.resize(m - 1, Real(0));
   }

   out.diag[0] = Real(1) / alpha_hist[0];

   for (Integer i = 1; i < m; ++i) {
      out.diag[i] = Real(1) / alpha_hist[i] + beta_hist[i - 1] / alpha_hist[i - 1];
      out.offdiag[i - 1] = std::sqrt(std::max(Real(0), beta_hist[i - 1])) / alpha_hist[i - 1];
   }

   if (static_cast<Integer>(beta_hist.size()) >= m) {
      out.trailing = std::sqrt(std::max(Real(0), beta_hist[m - 1])) / alpha_hist[m - 1];
   }

   return out;
}

/**
 * @brief Estimate spectral gaps between consecutive eigenvalues
 *
 * For each eigenvalue, computes the distance to its nearest neighbor(s).
 * Useful for understanding eigenvalue clustering and separation, which
 * affects CG convergence behavior.
 *
 * Large gaps indicate well-separated eigenvalues (good for CG convergence).
 * Small gaps indicate eigenvalue clusters (can slow CG convergence).
 *
 * @param vals Sorted eigenvalues in ascending order
 * @return Gap estimates: gaps[i] = min distance to left/right neighbor for vals[i]
 *         (infinity for singleton lists)
 */
inline
auto EstimateNeighborGaps(const std::vector<Real> & vals)
   -> std::vector<Real>
{
   const Integer n = static_cast<Integer>(vals.size());
   std::vector<Real> gaps(n, Real(0));

   if (n == 0) return gaps;
   if (n == 1) {
      gaps[0] = std::numeric_limits<Real>::infinity();
      return gaps;
   }

   for (Integer i = 0; i < n; ++i)
   {
      Real left  = std::numeric_limits<Real>::infinity();
      Real right = std::numeric_limits<Real>::infinity();

      if (i > 0)     left  = vals[i] - vals[i - 1];
      if (i + 1 < n) right = vals[i + 1] - vals[i];

      gaps[i] = std::min(left, right);
   }

   return gaps;
}

/**
 * @brief Ritz value estimates and associated error bounds
 *
 * Ritz values are eigenvalues of the Lanczos tridiagonal matrix T_k that
 * approximate the eigenvalues of the original matrix A. As CG progresses
 * (k increases), these approximations become more accurate.
 *
 * **Purpose:**
 * - Estimate condition number: cond(A) ≈ max(values) / min(values)
 * - Understand convergence rate: related to eigenvalue distribution
 * - Identify spectral gaps: regions with few eigenvalues
 * - Assess Ritz pair quality: via residual_estimates
 */
struct RitzEstimate
{
   std::vector<Real> values;              ///< Ritz values (eigenvalues of T_k, approximating eigenvalues of A)
   std::vector<Real> residual_estimates;  ///< ||A*z_i - theta_i*z_i|| estimates for each Ritz pair
   std::vector<Real> gap_estimates;       ///< Distance to nearest neighbor Ritz value
};

/**
 * @brief Compute Ritz values and residual estimates from Lanczos tridiagonal
 *
 * Computes the eigendecomposition of the Lanczos tridiagonal matrix T_k to obtain:
 * - **Ritz values**: Eigenvalues of T_k (approximate eigenvalues of A)
 * - **Residual estimates**: ||A*z_i - theta_i*z_i|| ≈ |trailing| * |last component of z_i|
 * - **Gap estimates**: Distance to neighboring Ritz values
 *
 * These estimates are used to:
 * - Compute condition number: max_ritz / min_ritz
 * - Assess convergence rate: related to eigenvalue distribution
 * - Identify well-approximated eigenvalues (small residual estimates)
 *
 * @param T Lanczos tridiagonal matrix snapshot from BuildLanczosFromCGCoefficients
 * @return RitzEstimate containing Ritz values, residual bounds, and gap estimates
 */
inline
auto EstimateRitzData(const LanczosTridiagonalSnapshot & T)
   -> RitzEstimate
{
   RitzEstimate out;

   const Integer m = static_cast<Integer>(T.diag.size());
   if (m == 0) return out;

   std::vector< std::vector<Real> > dense_T(m, std::vector<Real>(m, Real(0)));
   for (Integer i = 0; i < m; ++i) {
      dense_T[i][i] = T.diag[i];
   }
   for (Integer i = 0; i + 1 < m; ++i) {
      dense_T[i][i + 1] = T.offdiag[i];
      dense_T[i + 1][i] = T.offdiag[i];
   }

   const auto eig = SymmetricEigenDecompositionJacobi(dense_T);

   out.values = eig.eigenvalues;
   out.gap_estimates = EstimateNeighborGaps(out.values);
   out.residual_estimates.resize(m, Real(0));

   for (Integer i = 0; i < m; ++i) {
      // ||A z_i - theta_i z_i|| = trailing * | last component of s_i |
      const Real last_comp = eig.eigenvectors[i].empty() ? Real(0) : eig.eigenvectors[i].back();
      out.residual_estimates[i] = std::abs(T.trailing * last_comp);
   }

   return out;
}

// ------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------

/**
 * @brief Compute 2-norm of a vector using provided dot product
 *
 * @param dot Dot product function: dot(u, v) -> Real
 * @param v Vector to compute norm of
 * @return ||v|| = sqrt(v^T v)
 */
template <typename DotType, typename Vector>
inline
auto Norm2(const DotType & dot, const Vector & v) -> Real
{
   return std::sqrt(dot(v, v));
}

/**
 * @brief Return normalized copy of a vector
 *
 * @param dot Dot product function: dot(u, v) -> Real
 * @param v Vector to normalize
 * @return v / ||v|| (or zero vector if ||v|| = 0)
 */
template <typename DotType, typename Vector>
inline
auto NormalizedCopy(const DotType & dot, const Vector & v) -> Vector
{
   Vector out = v;
   const Real n = Norm2(dot, out);
   if (n > Real(0)) {
      out *= (Real(1) / n);
   }
   return out;
}

/**
 * @brief Measure loss of orthogonality between vector and history
 *
 * Computes max |<v, h>| / (||v|| ||h||) over all vectors h in history.
 * In exact arithmetic, CG residuals are mutually orthogonal, so this value
 * should be 0. Non-zero values indicate numerical loss of orthogonality.
 *
 * Values near 0: Good numerical behavior
 * Values near 1: Severe loss of orthogonality
 *
 * @param dot Dot product function: dot(u, v) -> Real
 * @param v Current vector
 * @param history Previous vectors that should be orthogonal to v
 * @return Maximum normalized inner product with history
 */
template <typename DotType, typename Vector>
inline
auto MaxNormalizedInnerProductWithHistory(
   const DotType & dot,
   const Vector & v,
   const std::vector<Vector> & history )
   -> Real
{
   if (history.empty()) return Real(0);

   const Real nv = Norm2(dot, v);
   if (nv == Real(0)) return Real(0);

   Real out = Real(0);
   for (const auto & h : history)
   {
      const Real nh = Norm2(dot, h);
      if (nh == Real(0)) continue;
      out = std::max(out, std::abs(dot(v, h)) / (nv * nh));
   }
   return out;
}

/**
 * @brief Measure loss of A-conjugacy between search directions
 *
 * Computes max |<p, A*p_j>| / sqrt(<p,A*p><p_j,A*p_j>) over all previous
 * search directions p_j in history.
 *
 * In exact arithmetic, CG search directions are A-conjugate (A-orthogonal),
 * meaning <p_i, A*p_j> = 0 for i ≠ j. Non-zero values indicate numerical
 * loss of conjugacy.
 *
 * Values near 0: Good numerical behavior
 * Values near 1: Severe loss of A-conjugacy
 *
 * @param dot Dot product function: dot(u, v) -> Real
 * @param p Current search direction
 * @param Ap A*p (operator applied to current search direction)
 * @param p_hist Previous search directions
 * @param Ap_hist A*p_j for each previous search direction
 * @return Maximum normalized A-inner product with history
 */
template <typename DotType, typename Vector>
inline
auto MaxAConjugacyWithHistory(
   const DotType & dot,
   const Vector & p,
   const Vector & Ap,
   const std::vector<Vector> & p_hist,
   const std::vector<Vector> & Ap_hist )
   -> Real
{
   if (p_hist.empty()) return Real(0);

   const Real self = dot(p, Ap);
   if (self <= Real(0)) return Real(0);

   Real out = Real(0);
   for (Integer j = 0; j < static_cast<Integer>(p_hist.size()); ++j)
   {
      const Real prev = dot(p_hist[j], Ap_hist[j]);
      if (prev <= Real(0)) continue;

      const Real denom = std::sqrt(self * prev);
      if (denom == Real(0)) continue;

      const Real val = std::abs(dot(p, Ap_hist[j])) / denom;
      out = std::max(out, val);
   }
   return out;
}

// ------------------------------------------------------------
// Host-side diagnostic CG
// ------------------------------------------------------------

template<
   typename Operator,
   typename RHSType,
   typename DotType,
   typename Vector
>
auto ConjugateGradientDiagnose(
   const Operator & A,
   const RHSType & rhs,
   const DotType & dot,
   const Integer max_iters,
   const Real tol,
   Vector & x,
   // scratch memory inputs (same role as in your original CG)
   Vector & tmp,
   Vector & z,
   Vector & residual,
   Vector & p,
   const CGDiagnosticOptions & options = {} )
   -> CGDiagnostics<Vector>
{
   CGDiagnostics<Vector> diag;

   // Extra local work vectors for diagnostics
   Vector true_residual( residual.Size() );
   true_residual = 0.0;
   Vector residual_gap( residual.Size() );
   residual_gap = 0.0;
   Vector Ax_x( residual.Size() );
   Ax_x = 0.0;

   std::vector<Real> alpha_hist;
   std::vector<Real> beta_hist;

   std::vector<Vector> residual_basis_hist; // normalized residuals
   std::vector<Vector> p_hist;
   std::vector<Vector> Ap_hist;

   // 1) r0 = rhs - A*x0
   A(x, tmp);
   residual = rhs;
   residual -= tmp;

   // Use actual ||rhs|| to match your documentation
   diag.rhs_norm = Norm2(dot, rhs);

   const Real r0_rec = Norm2(dot, residual);
   diag.initial_recursive_residual_norm = r0_rec;
   diag.initial_relative_recursive_residual =
      (diag.rhs_norm > Real(0)) ? (r0_rec / diag.rhs_norm) : r0_rec;

   if (options.recompute_true_residual_each_iter)
   {
      A(x, Ax_x);
      true_residual = rhs;
      true_residual -= Ax_x;

      diag.initial_true_residual_norm = Norm2(dot, true_residual);
      diag.initial_relative_true_residual =
         (diag.rhs_norm > Real(0)) ? (diag.initial_true_residual_norm / diag.rhs_norm)
                                   : diag.initial_true_residual_norm;
   }
   else
   {
      diag.initial_true_residual_norm = r0_rec;
      diag.initial_relative_true_residual = diag.initial_relative_recursive_residual;
   }

   // Trivial rhs = 0
   if (diag.rhs_norm == Real(0))
   {
      x = 0.0;
      diag.success = true;
      diag.iterations = 0;
      diag.relative_error = Real(0);
      return diag;
   }

   // Initial convergence
   if (r0_rec < tol * diag.rhs_norm)
   {
      diag.success = true;
      diag.iterations = 0;
      diag.relative_error = r0_rec / diag.rhs_norm;
      return diag;
   }

   // p0 = r0
   p = residual;
   Real rr_new = dot(residual, residual);

   if (options.store_basis && r0_rec > Real(0)) {
      residual_basis_hist.push_back(NormalizedCopy(dot, residual));
   }

   for (Integer iter = 0; iter < max_iters; ++iter)
   {
      // tmp = A * p_k
      A(p, tmp);

      const Real pAp = dot(p, tmp);
      if (pAp <= Real(0))
      {
         // CG breakdown / indefiniteness / severe FP damage
         const Real current_rel = Norm2(dot, residual) / diag.rhs_norm;
         diag.success = false;
         diag.iterations = iter;
         diag.relative_error = current_rel;
         return diag;
      }

      const Real rr_old = rr_new;
      const Real alpha = rr_old / pAp;

      // x_{k+1} = x_k + alpha p_k
      // r_{k+1} = r_k - alpha A p_k
      Axpy( alpha, p, x );
      Axpy(-alpha, tmp, residual );

      const Real rec_r_norm = Norm2(dot, residual);

      // z = r (unpreconditioned case)
      z = residual;
      rr_new = dot(residual, z);

      const Real beta = (rr_old > Real(0)) ? (rr_new / rr_old) : Real(0);

      CGIterationDiagnostics<Vector> step;
      step.iter   = iter + 1;
      step.alpha  = alpha;
      step.beta   = beta;
      step.pAp    = pAp;
      step.rr_old = rr_old;
      step.rr_new = rr_new;

      step.recursive_residual_norm = rec_r_norm;
      step.relative_recursive_residual = rec_r_norm / diag.rhs_norm;

      step.x_norm  = Norm2(dot, x);
      step.p_norm  = Norm2(dot, p);
      step.Ap_norm = Norm2(dot, tmp);

      if (options.recompute_true_residual_each_iter)
      {
         A(x, Ax_x);
         true_residual = rhs;
         true_residual -= Ax_x;

         residual_gap = residual;
         residual_gap -= true_residual;

         step.true_residual_norm = Norm2(dot, true_residual);
         step.relative_true_residual = step.true_residual_norm / diag.rhs_norm;
         step.residual_gap_norm = Norm2(dot, residual_gap);

         step.xAx = dot(x, Ax_x);
         step.bTx = dot(rhs, x);
         step.objective = Real(0.5) * step.xAx - step.bTx;
      }
      else
      {
         step.true_residual_norm = step.recursive_residual_norm;
         step.relative_true_residual = step.relative_recursive_residual;
         step.residual_gap_norm = Real(0);
      }

      if (options.store_basis)
      {
         step.max_residual_orthogonality =
            MaxNormalizedInnerProductWithHistory(
               dot,
               residual,
               residual_basis_hist );

         step.max_search_A_conjugacy =
            MaxAConjugacyWithHistory(
               dot,
               p,
               tmp,
               p_hist,
               Ap_hist );

         if (rec_r_norm > Real(0)) {
            residual_basis_hist.push_back(NormalizedCopy(dot, residual));
         }

         p_hist.push_back(p);
         Ap_hist.push_back(tmp);
      }

      // Recover T_{k+1} and Ritz data
      alpha_hist.push_back(alpha);
      beta_hist.push_back(beta);

      if (options.estimate_ritz)
      {
         const auto T = BuildLanczosFromCGCoefficients(alpha_hist, beta_hist, iter + 1);

         step.lanczos_diag     = T.diag;
         step.lanczos_offdiag  = T.offdiag;
         step.lanczos_trailing = T.trailing;

         const auto ritz = EstimateRitzData(T);
         step.ritz_values              = ritz.values;
         step.ritz_residual_estimates  = ritz.residual_estimates;
         step.ritz_gap_estimates       = ritz.gap_estimates;

         if (!step.ritz_values.empty()) {
            step.min_ritz = step.ritz_values.front();
            step.max_ritz = step.ritz_values.back();
            if (step.min_ritz > Real(0)) {
               step.estimated_condition = step.max_ritz / step.min_ritz;
            }
         }
      }

      if (options.store_snapshots)
      {
         step.x_snapshot = Vector(x);
         step.residual_snapshot = Vector(residual);
         step.p_snapshot = Vector(p);
      }

      diag.steps.push_back(std::move(step));

      // Convergence test
      if (rec_r_norm < tol * diag.rhs_norm)
      {
         diag.success = true;
         diag.iterations = iter + 1;
         diag.relative_error = rec_r_norm / diag.rhs_norm;
         return diag;
      }

      // p_{k+1} = r_{k+1} + beta p_k
      p *= beta;
      p += z;
   }

   const Real final_rel = Norm2(dot, residual) / diag.rhs_norm;
   diag.success = false;
   diag.iterations = max_iters;
   diag.relative_error = final_rel;
   return diag;
}

} // namespace gendil
