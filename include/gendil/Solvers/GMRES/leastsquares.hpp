// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Compute GMRES correction coefficients from the small Arnoldi problem.
 *
 * After each new basis direction, GMRES minimizes
 * \f$\|\beta e_0 - H y\|_2\f$: H holds the Arnoldi coefficients and beta is the
 * cycle's initial residual norm. Givens rotations progressively reduce H to
 * upper-triangular form. Back substitution then finds the coefficients y used
 * to combine the search directions into a solution correction.
 *
 * InitializeReducedLeastSquares starts a cycle, UpdateReducedQR incorporates
 * each new column, and BackSubstitute solves the accepted reduced system.
 * The QR residual estimate helps decide when to end a cycle; the original-system
 * residual remains the solver's convergence test.
 *
 * Serial overloads act on private arrays. Native-context overloads require
 * uniform participation, publish leader-written shared data, and return the
 * same status and scalar results to every participant. Buffers must already
 * have the active dimensions; these numerical helpers do not allocate or
 * validate storage capacity.
 */
#include "gendil/Algebra/Vector/localoperations.hpp"
#include "gendil/Utilities/MathHelperFunctions/hypot.hpp"
#include "gendil/Utilities/MathHelperFunctions/isfinite.hpp"
#include <limits>

namespace gendil::gmres_detail
{
/// Whether a new reduced column yields a usable rotation and residual estimate.
enum class ReducedStatus
{
   /// A finite, nonzero rotation radius permits the reduced update.
   Usable,
   /// Both entries defining the new rotation are zero.
   Singular,
   /// A column entry or a computed rotation/RHS value is not finite.
   NonFinite
};
/// Outcome of one reduced QR update; no shared storage is borrowed by the result.
struct ReducedQRResult
{
   ReducedStatus status;
   Real residual_norm_estimate = 0; // Valid only when status == Usable.
};

namespace completion_detail
{

/// Insert the new remainder norm and apply earlier rotations to column j.
template <class H, class Cosines, class Sines>
GENDIL_HOST_DEVICE void ApplyPreviousRotations(H &h, const Cosines &c, const Sines &s,
                                               Integer j, Real next_norm)
{
   h(j + 1, j) = next_norm;
   for (Integer i = 0; i < j; ++i)
   {
      const Real upper = h(i, j), lower = h(i + 1, j);
      h(i, j) = c[i] * upper + s[i] * lower;
      h(i + 1, j) = -s[i] * upper + c[i] * lower;
   }
}

/// Candidate rotation and transformed RHS, retained until all values are valid.
struct Rotation
{
   ReducedStatus status;
   Real radius = 0, cosine = 0, sine = 0, rhs_upper = 0, rhs_lower = 0;
};

// Validate the column and compute a candidate without modifying shared storage.
// Every participant can derive the same result from the published operands.
template <class H, class Rhs>
GENDIL_HOST_DEVICE Rotation ComputeRotation(const H &h, const Rhs &g, Integer j)
{
   for (Integer i = 0; i <= j + 1; ++i)
   {
      if (!IsFinite(h(i, j)))
      {
         return {ReducedStatus::NonFinite};
      }
   }
   const Real upper = h(j, j), lower = h(j + 1, j);
   const Real radius = gendil::Hypot(upper, lower);
   if (!IsFinite(radius))
   {
      return {ReducedStatus::NonFinite};
   }
   if (radius == 0)
   {
      return {ReducedStatus::Singular};
   }
   const Real cosine = upper / radius, sine = lower / radius;
   const Real rhs_upper = cosine * g[j] + sine * g[j + 1];
   const Real rhs_lower = -sine * g[j] + cosine * g[j + 1];
   if (!IsFinite(cosine) || !IsFinite(sine) || !IsFinite(rhs_upper) ||
       !IsFinite(rhs_lower))
   {
      return {ReducedStatus::NonFinite};
   }
   return {ReducedStatus::Usable, radius, cosine, sine, rhs_upper, rhs_lower};
}

/// Commit a validated rotation, eliminate the subdiagonal, and transform g.
template <class H, class Cosines, class Sines, class Rhs>
GENDIL_HOST_DEVICE void CommitRotation(H &h, Cosines &c, Sines &s, Rhs &g, Integer j,
                                       const Rotation &rotation)
{
   c[j] = rotation.cosine;
   s[j] = rotation.sine;
   h(j, j) = rotation.radius;
   h(j + 1, j) = 0;
   g[j] = rotation.rhs_upper;
   g[j + 1] = rotation.rhs_lower;
}

/// One candidate back-substitution coefficient, not yet written to y.
struct Coefficient
{
   bool valid;
   Real value;
};

/// Solve one row using the already-computed higher-index entries of y.
template <class H, class Rhs, class Coefficients>
GENDIL_HOST_DEVICE Coefficient ComputeCoefficient(const H &h, const Rhs &g,
                                                  const Coefficients &y, Integer row,
                                                  Integer count)
{
   Real sum = g[row];
   const Real diagonal = h(row, row);
   if (!IsFinite(sum) || !IsFinite(diagonal) || diagonal == 0)
   {
      return {false, 0};
   }
   for (Integer j = row + 1; j < count; ++j)
   {
      sum -= h(row, j) * y[j];
      if (!IsFinite(sum))
      {
         return {false, 0};
      }
   }
   const Real value = sum / diagonal;
   return {IsFinite(value), value};
}
} // namespace completion_detail

/// Write one projection coefficient into its reduced-system storage.
template <class Coefficients>
GENDIL_HOST_DEVICE void StoreCoefficient(Coefficients &c, Integer i, Real value)
{
   c[i] = value;
}

/// Let the primary participant write the coefficient and publish it before return.
template <class Context, class Coefficients>
GENDIL_HOST_DEVICE void StoreCoefficient(Context &context, Coefficients &c, Integer i,
                                         Real value)
{
   if (IsPrimaryThread(context))
   {
      StoreCoefficient(c, i, value);
   }
   if constexpr (gendil::is_threaded_v<Context>)
   {
      context.Synchronize();
   }
}

/** Start one cycle with identity rotations, zero coefficients, and g = beta e_0.
 * c, s, and y need m entries; g needs m+1. Only those active prefixes are
 * initialized. Hessenberg columns are supplied later by Arnoldi.
 */
template <class Cosines, class Sines, class Rhs, class Coefficients>
GENDIL_HOST_DEVICE void InitializeReducedLeastSquares(Cosines &c, Sines &s, Rhs &g,
                                                      Coefficients &y, Integer m,
                                                      Real beta)
{
   for (Integer i = 0; i < m; ++i)
   {
      c[i] = 1;
      s[i] = 0;
      g[i] = 0;
      y[i] = 0;
   }
   g[m] = 0;
   g[0] = beta;
}

/// Initialize the shared reduced arrays on the primary participant and publish them.
template <class Context, class Cosines, class Sines, class Rhs, class Coefficients>
GENDIL_HOST_DEVICE void InitializeReducedLeastSquares(Context &context, Cosines &c,
                                                      Sines &s, Rhs &g, Coefficients &y,
                                                      Integer m, Real beta)
{
   if (IsPrimaryThread(context))
   {
      InitializeReducedLeastSquares(c, s, g, y, m, beta);
   }
   if constexpr (gendil::is_threaded_v<Context>)
   {
      context.Synchronize();
   }
}

/** Incorporate column j and return its updated residual-norm estimate.
 * H(0:j,j) contains the new projections; next_norm supplies H(j+1,j).
 * Earlier rotations in c and s must already be available. A usable update
 * overwrites this column, c[j], s[j], g[j], and g[j+1].
 *
 * The estimate is meaningful only for Usable. On Singular or NonFinite,
 * earlier rotations have still modified this column; the update is not rolled
 * back. H needs rows through j+1, and g needs entries through j+1.
 */
template <class H, class Cosines, class Sines, class Rhs>
GENDIL_HOST_DEVICE ReducedQRResult UpdateReducedQR(H &h, Cosines &c, Sines &s, Rhs &g,
                                                   Integer j, Real next_norm)
{
   completion_detail::ApplyPreviousRotations(h, c, s, j, next_norm);
   const auto rotation = completion_detail::ComputeRotation(h, g, j);
   if (rotation.status == ReducedStatus::Usable)
   {
      completion_detail::CommitRotation(h, c, s, g, j, rotation);
   }
   return {rotation.status, rotation.status == ReducedStatus::Usable
                                ? std::abs(rotation.rhs_lower)
                                : Real(0)};
}

/** Perform the QR update collectively, including its shared read/write boundaries.
 * Complete prior column reads before mutation, publish the rotated column, and
 * finish peer reads before committing the new rotation. The returned estimate
 * is a replicated value, so callers need not reread a shared RHS entry.
 */
template <class Context, class H, class Cosines, class Sines, class Rhs>
GENDIL_HOST_DEVICE ReducedQRResult UpdateReducedQR(Context &context, H &h, Cosines &c,
                                                   Sines &s, Rhs &g, Integer j,
                                                   Real next_norm)
{
   // The preceding orthogonalization may still have readers of this column.
   // A supplied norm has no obligation to complete those unrelated H accesses.
   if constexpr (gendil::is_threaded_v<Context>)
   {
      context.Synchronize();
   }
   if (IsPrimaryThread(context))
   {
      completion_detail::ApplyPreviousRotations(h, c, s, j, next_norm);
   }
   if constexpr (gendil::is_threaded_v<Context>)
   {
      context.Synchronize();
   }
   const auto rotation = completion_detail::ComputeRotation(h, g, j);
   if constexpr (gendil::is_threaded_v<Context>)
   {
      context.Synchronize();
   }
   if (rotation.status == ReducedStatus::Usable)
   {
      if (IsPrimaryThread(context))
      {
         completion_detail::CommitRotation(h, c, s, g, j, rotation);
      }
      if constexpr (gendil::is_threaded_v<Context>)
      {
         context.Synchronize();
      }
   }
   // Estimate is valid only for Usable. It is replicated, not a later shared read.
   return {rotation.status, rotation.status == ReducedStatus::Usable
                                ? std::abs(rotation.rhs_lower)
                                : Real(0)};
}

/** Solve the leading count-by-count triangular system H y = g.
 * H and g are read-only; y receives the correction coefficients. Return false
 * for a zero/nonfinite pivot or a nonfinite computed coefficient. Already
 * written higher-index coefficients remain, so use y only after a true result.
 */
template <class H, class Rhs, class Coefficients>
GENDIL_HOST_DEVICE bool BackSubstitute(const H &h, const Rhs &g, Coefficients &y,
                                       Integer count)
{
   for (Integer row = count; row-- > 0;)
   {
      const auto coefficient = completion_detail::ComputeCoefficient(h, g, y, row, count);
      if (!coefficient.valid)
      {
         return false;
      }
      y[row] = coefficient.value;
   }
   return true;
}

/** Back-substitute collectively and publish each coefficient before its next use.
 * All participants read the same reduced system and return the same success
 * value. A failed coefficient is not committed; earlier writes are retained.
 */
template <class Context, class H, class Rhs, class Coefficients>
GENDIL_HOST_DEVICE bool BackSubstitute(Context &context, const H &h, const Rhs &g,
                                       Coefficients &y, Integer count)
{
   for (Integer row = count; row-- > 0;)
   {
      const auto coefficient = completion_detail::ComputeCoefficient(h, g, y, row, count);
      if (!coefficient.valid)
      {
         if constexpr (gendil::is_threaded_v<Context>)
         {
            context.Synchronize();
         }
         return false;
      }
      // This row only reads y[j] for j > row: no peer reads the overwritten slot.
      if (IsPrimaryThread(context))
      {
         y[row] = coefficient.value;
      }
      if constexpr (gendil::is_threaded_v<Context>)
      {
         context.Synchronize();
      }
   }
   return true;
}

} // namespace gendil::gmres_detail
