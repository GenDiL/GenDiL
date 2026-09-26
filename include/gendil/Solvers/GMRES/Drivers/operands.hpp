// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/**
 * @file
 * @brief Check logical GMRES operands and form their Hessenberg column views.
 *
 * Checks use the operands' declared dimensions, not hidden allocation lengths.
 * Column views preserve the supplied origin and strides without copying entries.
 */

#include "gendil/Algebra/Vector/localoperations.hpp"
#include "gendil/Solvers/GMRES/arnoldi.hpp"
#include "gendil/Solvers/GMRES/options.hpp"
#include "gendil/Solvers/GMRES/preconditioning.hpp"
#include "gendil/Utilities/View/Containers/referencecontainer.hpp"
#include "gendil/Utilities/View/Layouts/offsetlayout.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include <array>
#include <cstddef>

namespace gendil::gmres_detail
{

/** Borrow column j for orthogonalization without copying Hessenberg entries.
 * Entry i retains the address of h(i, j), including its origin and row stride.
 * The view has h's declared row extent and borrows the h.data object itself;
 * that object and its underlying storage must remain valid while it is used.
 */
template <class H> GENDIL_HOST_DEVICE auto MakeHessenbergColumnView(H &h, Integer j)
{
   return gendil::MakeView(
       gendil::ReferenceContainer{h.data},
       gendil::MakeOffsetLayout(
           gendil::StridedLayout<1, gendil::DynamicExtents<1>>{
               std::array<gendil::GlobalIndex, 1>{gendil::GetStride<0>(h.layout)},
               gendil::DynamicExtents<1>{gendil::GetExtentValue<0>(h)}},
           h.layout.Offset(0, j)));
}

/** Return the full vector length used to compare local solve operands.
 * A threaded view contributes its collective logical shape, not the number of
 * registers held by one participant; other operands use their local size.
 */
template <class V> GENDIL_HOST_DEVICE size_t LogicalSize(const V &v)
{
   if constexpr (requires { typename V::logical_extents_type; })
      return gendil::GetLogicalSize(typename V::logical_extents_type{});
   else
      return LocalSize(v);
}

/** Check declared operands for an n-entry solve with active restart length m.
 * Only the active basis, dense prefixes, and selected preconditioning directions
 * are checked. Their capacities may differ; no common reservation is inferred.
 * This does not inspect backing allocations, aliasing, or memory accessibility.
 */
template <PreconditioningMode mode, class B>
GENDIL_HOST_DEVICE bool ValidBuffers(B &buffers, size_t n, Integer m)
{
   if (m >= buffers.basis.Size() || m >= gendil::GetExtentValue<0>(buffers.hessenberg) ||
       m > gendil::GetExtentValue<1>(buffers.hessenberg) || m > buffers.cosines.Size() ||
       m > buffers.sines.Size() || m > buffers.coefficients.Size() ||
       m >= buffers.reduced_rhs.Size() || LogicalSize(buffers.work) != n)
      return false;
   for (Integer j = 0; j <= m; ++j)
   {
      auto &&v = buffers.basis(j);
      if (LogicalSize(v) != n)
         return false;
   }
   if constexpr (mode != PreconditioningMode::None)
   {
      const size_t d = mode == PreconditioningMode::FixedRight ? 1 : m;
      if (d > buffers.directions.Size())
         return false;
      for (size_t j = 0; j < d; ++j)
      {
         auto &&v = buffers.directions(j);
         if (LogicalSize(v) != n)
            return false;
      }
   }
   return true;
}

/// Require a positive restart and finite, nonnegative tolerances; a zero budget is valid.
inline bool ValidOptions(const GMRESOptions &options)
{
   return options.restart > 0 && ValidNorm(options.relative_tolerance) &&
          ValidNorm(options.absolute_tolerance);
}

} // namespace gendil::gmres_detail
