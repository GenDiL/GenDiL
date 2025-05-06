// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include "faceconnectivity.hpp"

namespace gendil {

template < typename Geometry >
struct ConformingCellConnectivity
{
   static constexpr Integer dim = Geometry::space_dim;
   using OrientationType = Permutation< dim >; // !FIXME: Only true for hypercube geometries?
   using BoundaryType = bool;
   using FaceInfo = FaceConnectivity< 0, Geometry, OrientationType, BoundaryType, Empty >;
   FaceInfo faces[ Geometry::num_faces ];
};

}