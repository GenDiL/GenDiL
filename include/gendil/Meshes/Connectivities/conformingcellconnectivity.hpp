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
   using local_face_index_type = Empty;
   using OrientationType = Permutation< dim >; // !FIXME: Only true for hypercube geometries?
   using normal_type = Empty;
   using BoundaryType = bool;
   using FaceInfo = FaceView< local_face_index_type, Geometry, OrientationType, normal_type, ConformingFaceMap<dim>, BoundaryType >;
   FaceInfo faces[ Geometry::num_faces ];
};

}