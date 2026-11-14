// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/facetnormal.hpp"

namespace gendil
{

// =============================================================================
// Normal expression
// =============================================================================
//
// Returns the physical unit normal supplied by generic facet execution.
// Both selected interior sides store the common canonical normal; boundary
// normals remain outward. GetFacetNormal reads the stored value unchanged.
// Custom contexts can supply a free GetFacetNormal overload with the same convention.
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
      return GetFacetNormal(face_context, quad_pt_context);
   }
};

inline std::ostream& operator<<(std::ostream& os, const Normal& normal)
{
   return os << "n";
}

} // namespace gendil
