// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil
{

// Point construction supplies the shared canonical unit normal on interior
// facets and the outward unit normal on physical boundaries.
template<class FacetContext, class PointContext>
   requires requires(const PointContext& point) { point.physical_normal; }
GENDIL_HOST_DEVICE
constexpr auto GetFacetNormal(const FacetContext&, const PointContext& point)
{
   return point.physical_normal;
}

} // namespace gendil
