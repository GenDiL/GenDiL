// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

namespace gendil
{

// =============================================================================
// Normal expression
// =============================================================================
//
// Returns the physical unit normal supplied by generic facet execution.
// Boundary, local-interior, and global-interior contexts obtain it from
// ComputeFacetGeometry. Both selected interior sides use the common normal.
// A compatibility fallback for contexts without physical_normal still returns
// unnormalized J^{-T} n_ref; generic facet execution does not use that fallback.
//
struct Normal : FieldBase
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename FaceContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext & kernel_context,
      const WeakFormContext & weak_form_context,
      const OperatorContext & operator_context,
      const FaceContext & face_context,
      const QuadPtContext & quad_pt_context,
      const Fields & fields ) const
   {
      // Check if quad_pt_context has physical_normal field (FacetQuadraturePointContext)
      if constexpr (requires { quad_pt_context.physical_normal; })
      {
         // Facet geometry has already normalized this normal.
         return quad_pt_context.physical_normal;
      }
      else
      {
         // Compatibility path for callers without facet geometry.
         const auto reference_normal = GetReferenceNormal( face_context );
         const auto physical_normal = ComputePhysicalNormal( quad_pt_context.inv_J_mesh, reference_normal );
         return physical_normal;
      }
   }
};

inline std::ostream& operator<<(std::ostream& os, const Normal& normal)
{
   return os << "n";
}

} // namespace gendil
