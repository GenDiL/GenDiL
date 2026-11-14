// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include <cstdint>
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
   // Zero retains the lower-cell-index convention for manually built meshes;
   // importers may record their own shared side ordering with explicit +/-1.
   using normal_orientation_type = std::int8_t;
   using FaceInfo = FaceView< local_face_index_type, Geometry, OrientationType,
      normal_type, ConformingFaceMap<dim>, BoundaryType, normal_orientation_type >;
   FaceInfo faces[ Geometry::num_faces ];
};

}
