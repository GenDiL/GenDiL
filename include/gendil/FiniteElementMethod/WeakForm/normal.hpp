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
// Returns physical normal for facet integrands.
//
// **Behavior:**
//   - With FacetQuadraturePointContext: returns normalized physical_normal field
//   - With regular QuadraturePointContext: returns old ComputePhysicalNormal (unnormalized)
//
// **Rationale:**
//   FacetQuadraturePointContext stores the cleaned-up normalized physical normal
//   computed via ComputeFacetGeometry. Old production path uses unnormalized normal.
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
         // Use cleaned-up normalized physical normal from FacetQuadraturePointContext
         return quad_pt_context.physical_normal;
      }
      else
      {
         // Old production path: compute unnormalized physical normal
         const auto reference_normal = GetReferenceNormal( face_context );
         const auto physical_normal = ComputePhysicalNormal( quad_pt_context.inv_J_mesh, reference_normal );
         return physical_normal;
      }
   }
};

std::ostream& operator<<(std::ostream& os, const Normal& normal)
{
   return os << "n";
}

} // namespace gendil
