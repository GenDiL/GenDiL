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

enum FaceType { Interior, Distributed, Boundary };

// TODO: Specialize for FaceConnectivity?
template < typename FaceInfo >
constexpr GlobalIndex GetNeighborIndex( const FaceInfo & face_info )
{
   return face_info.neighbor_index;
}

constexpr bool IsBoundaryFace( const bool & face_info )
{
   return face_info;
}

template < bool val >
constexpr bool IsBoundaryFace( const std::integral_constant<bool, val> & face_info )
{
   return val;
}

constexpr bool IsBoundaryFace( const FaceType & face_info )
{
   return face_info == Boundary;
}

// TODO: Replace with explicit type FaceConnectivity?
template < typename FaceInfo >
constexpr bool IsBoundaryFace( const FaceInfo & face_info )
{
   return IsBoundaryFace( face_info.boundary );
}

constexpr bool IsDistributedFace( const bool & face_info )
{
   return false;
}

constexpr bool IsDistributedFace( const std::integral_constant<bool, false> & face_info )
{
   return false;
}

constexpr bool IsDistributedFace( const FaceType & face_info )
{
   return face_info == Distributed;
}

template < typename FaceInfo >
constexpr bool IsDistributedFace( const FaceInfo & face_info )
{
   return IsDistributedFace( face_info.boundary );
}

template < typename FaceInfo >
GENDIL_HOST_DEVICE
constexpr auto GetReferenceNormal( const FaceInfo & face_info )
{
   return face_info.normal;
}

}
