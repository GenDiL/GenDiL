// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontexttraits.hpp"

namespace gendil
{

// Select an interior trace from already canonical point data. Local
// current/neighbor data uses the overload beside LocalFacetContext instead.
template<class FacetContext, class PointContext>
   requires (!is_physical_boundary_v<PointContext> &&
      requires(const PointContext& point) { point.MinusSide(); } &&
      !requires(const PointContext& point) { point.inv_J_mesh_current; } &&
      !requires(const FacetContext& facet)
      {
         facet.CurrentSide();
         facet.NeighborSide();
      })
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetMinusSide(const FacetContext&, const PointContext& point)
{
   return point.MinusSide();
}

template<class FacetContext, class PointContext>
   requires (!is_physical_boundary_v<PointContext> &&
      requires(const PointContext& point) { point.PlusSide(); } &&
      !requires(const PointContext& point) { point.inv_J_mesh_current; } &&
      !requires(const FacetContext& facet)
      {
         facet.CurrentSide();
         facet.NeighborSide();
      })
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetPlusSide(const FacetContext&, const PointContext& point)
{
   return point.PlusSide();
}

} // namespace gendil
