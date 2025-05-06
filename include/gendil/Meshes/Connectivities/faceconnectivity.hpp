// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "orientation.hpp"

namespace gendil{

// !FIXME: This assumes conforming and same topologies on each side
template <
   Integer LocalFaceIndex, // TODO: Generalize for AMR? Should we replace with std::integral_constant?
   typename Geometry, // TODO: Generalize, this currently only describe the volume element (assumes the two sides are the same)
   // typename ConformityType,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType >
struct FaceConnectivity
{
   using geometry = Geometry;
   // using conformity_type = ConformityType;
   using orientation_type = OrientationType;
   using boundary_type = BoundaryType;
   using normal_type = NormalType;

   static constexpr Integer dim = geometry::geometry_dim;
   static constexpr Integer local_face_index = LocalFaceIndex;
   static constexpr Integer neighbor_local_face_index = LocalFaceIndex < dim ? LocalFaceIndex + dim : LocalFaceIndex - dim; // TODO: This feels like magic / Only true for hypercubes => Should be Geometry function

   GlobalIndex neighbor_index;
   // conformity_type conformity;
   orientation_type orientation;
   boundary_type boundary;
   normal_type normal;
};

// TODO: Specialize for FaceConnectivity?
template < typename FaceInfo >
constexpr GlobalIndex GetNeighborIndex( const FaceInfo & face_info )
{
   return face_info.neighbor_index;
}

template < typename FaceInfo >
constexpr bool IsBoundaryFace( const FaceInfo & face_info )
{
   return face_info.boundary;
}

template < typename FaceInfo >
GENDIL_HOST_DEVICE
constexpr auto GetReferenceNormal( const FaceInfo & face_info )
{
   return face_info.normal;
}

}
