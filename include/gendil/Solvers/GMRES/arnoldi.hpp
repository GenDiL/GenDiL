// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/** @file @brief Find a new search direction orthogonal to the existing Krylov basis.
 *
 * Arnoldi applies the system to a search direction, then removes its components
 * along the existing basis. The projection coefficients form a Hessenberg
 * matrix column; the remaining vector can become the next basis vector.
 *
 * Modified Gram–Schmidt projects and subtracts one component at a time.
 * Two-pass classical Gram–Schmidt projects against the whole basis before
 * subtracting, then repeats to improve orthogonality. The restart algorithm,
 * not these helpers, normalizes the remainder and decides whether it is usable.
 *
 * Callbacks are borrowed, including mutable/noncopyable objects. Each completes
 * its advertised outputs before returning. Collective callers participate
 * uniformly and obtain identical scalar results; coefficient publication is
 * the responsibility of the supplied store or accumulation operation.
 */
#include "gendil/Solvers/GMRES/options.hpp"
#include "gendil/Solvers/iterativesolverutilities.hpp"
#include "gendil/Utilities/MathHelperFunctions/isfinite.hpp"

namespace gendil
{
/// Select sequential projection and subtraction during Arnoldi orthogonalization.
struct ModifiedGramSchmidt
{
};
/// Select two classical projection/update passes to improve basis orthogonality.
struct ClassicalGramSchmidtTwoPass
{
};
namespace gmres_detail
{
/** Remove count orthonormal basis components from work and store them in h.
 * Each projection uses the updated remainder. Neither the basis nor its
 * vectors are copied. Returns false on a nonfinite coefficient; earlier
 * coefficient writes and work updates remain in place.
 */
template <class Dot, class Axpy, class Store, class Basis, class Work, class H>
GENDIL_HOST_DEVICE bool OrthogonalizeModifiedGramSchmidt(
    Dot &dot, Axpy &axpy, Store &store, Basis &basis, Integer count, Work &work, H &h)
{
   for (Integer j = 0; j < count; ++j)
   {
      auto &&v = basis(j);
      const Real value = dot(v, work);
      if (!IsFinite(value))
      {
         return false;
      }
      store(h, j, value); // This operation publishes its own coefficient write.
      axpy(-value, v, work);
   }
   return true;
}

/** Orthogonalize work with two classical passes and sum their coefficients.
 * project reads an unchanged work vector before combine subtracts that pass's
 * basis combination. projections receives the total; separate corrections
 * scratch holds the second pass. Each coefficient buffer has count entries.
 * Returns false on nonfinite pass or accumulated coefficients, without rollback.
 */
template <class Project, class Combine, class AccumulateCoefficients, class Basis,
          class Work, class Projections, class Corrections>
GENDIL_HOST_DEVICE bool OrthogonalizeClassicalGramSchmidtTwoPass(
    Project &project, Combine &combine,
    AccumulateCoefficients &accumulate_coefficients, Basis &basis, Integer count,
    Work &work, Projections &projections, Corrections &corrections)
{
   project(basis, count, work, projections);
   for (Integer j = 0; j < count; ++j)
   {
      if (!IsFinite(projections[j]))
      {
         return false;
      }
   }

   combine(basis, count, projections, Real(-1), work);
   project(basis, count, work, corrections);
   for (Integer j = 0; j < count; ++j)
   {
      if (!IsFinite(corrections[j]))
      {
         return false;
      }
   }

   combine(basis, count, corrections, Real(-1), work);
   accumulate_coefficients(projections, corrections, count);
   for (Integer j = 0; j < count; ++j)
   {
      if (!IsFinite(projections[j]))
      {
         return false;
      }
   }
   return true;
}

/// Norms used by the restart algorithm to recognize a dependent new direction.
struct ArnoldiColumnResult
{
   bool valid;
   Real norm_before_orthogonalization;
   Real orthogonal_remainder_norm;
};
/** Expand one column and leave its unnormalized orthogonal remainder in work.
 * expand(column) writes work; orthogonalize(column) removes the existing basis
 * components and records coefficients. The supplied norm is evaluated before
 * and after orthogonalization. An invalid norm or failed orthogonalization
 * yields valid == false; partially updated scratch is not restored.
 */
template <class Work, class Expand, class Orthogonalize, class Norm>
GENDIL_HOST_DEVICE ArnoldiColumnResult BuildArnoldiColumn(Work &work, Integer column,
                                                          Expand &expand,
                                                          Orthogonalize &orthogonalize,
                                                          Norm &norm)
{
   expand(column);
   const Real norm_before_orthogonalization = norm(work);
   if (!solver_detail::ValidNorm(norm_before_orthogonalization) || !orthogonalize(column))
      return {false, norm_before_orthogonalization, 0};
   const Real orthogonal_remainder_norm = norm(work);
   return {solver_detail::ValidNorm(orthogonal_remainder_norm), norm_before_orthogonalization,
           orthogonal_remainder_norm};
}

} // namespace gmres_detail
} // namespace gendil
