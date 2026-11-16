// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontexttraits.hpp"

#include <type_traits>

namespace gendil
{

// Read an unmodified inverse Jacobian for a coefficient callback.
// Local interior geometry uses the overload beside LocalFacetContext so its
// callback reference frame cannot be bypassed by the raw-storage fallback.
template<class Context, class PointContext>
   requires (requires(const PointContext& point) { point.inv_J_mesh; } &&
      (is_physical_boundary_v<PointContext> ||
       !requires(const Context& context)
      {
         context.CurrentSide();
         context.NeighborSide();
      }))
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetCoefficientJacobianInverse(
   const Context&, PointContext& point)
{
   const auto& read_point = point;
   return (read_point.inv_J_mesh);
}

// A temporary point must not leave a reference to its native Jacobian storage.
// Named points retain the reference fast path used during coefficient evaluation.
template<class Context, class PointContext>
   requires (!std::is_lvalue_reference_v<PointContext> &&
      requires(const Context& context, const PointContext& point)
      {
         GetCoefficientJacobianInverse(context, point);
      })
GENDIL_HOST_DEVICE
constexpr auto GetCoefficientJacobianInverse(
   const Context& context, PointContext&& point)
{
   const auto& read_point = point;
   return GetCoefficientJacobianInverse(context, read_point);
}

} // namespace gendil
