// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>
#include <tuple>
#include <utility>
#include <vector>

#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructuredglobalfacetuples.hpp"

namespace gendil {
namespace unstructured_global_face_detail {

// Transitional vector-bucket materialization used by current builders and the
// restricted local-connectivity fallback. Stage B replaces this with exact-size
// storage and removes or reduces this header.
template < typename Geometry >
using InteriorBuckets =
   std::array<
      std::vector< UnstructuredInteriorFaceRecord< Geometry::geometry_dim > >,
      Geometry::num_faces >;

template < typename Geometry >
using BoundaryBuckets =
   std::array< std::vector< UnstructuredBoundaryFaceRecord >, Geometry::num_faces >;

template < typename Geometry >
using NonconformingInteriorBuckets =
   std::array<
      std::vector< UnstructuredNonconformingInteriorFaceRecord< Geometry::geometry_dim > >,
      Geometry::num_faces >;

template < typename Geometry, Integer... I >
auto MakeUnstructuredInteriorFaceConnectivity(
   const InteriorBuckets< Geometry > & buckets,
   std::integer_sequence< Integer, I... > )
{
   return std::make_tuple(
      UnstructuredInteriorFaceConnectivity< Geometry, I >( buckets[ I ] )... );
}

template < typename Geometry >
auto MakeUnstructuredInteriorFaceConnectivity(
   const InteriorBuckets< Geometry > & buckets )
{
   return MakeUnstructuredInteriorFaceConnectivity< Geometry >(
      buckets,
      std::make_integer_sequence< Integer, Geometry::num_faces >{} );
}

template < typename Geometry, Integer... I >
auto MakeUnstructuredNonconformingInteriorFaceConnectivity(
   const NonconformingInteriorBuckets< Geometry > & buckets,
   std::integer_sequence< Integer, I... > )
{
   return std::make_tuple(
      UnstructuredNonconformingInteriorFaceConnectivity< Geometry, I >( buckets[ I ] )... );
}

template < typename Geometry >
auto MakeUnstructuredNonconformingInteriorFaceConnectivity(
   const NonconformingInteriorBuckets< Geometry > & buckets )
{
   return MakeUnstructuredNonconformingInteriorFaceConnectivity< Geometry >(
      buckets,
      std::make_integer_sequence< Integer, Geometry::num_faces >{} );
}

template < typename Geometry, Integer... I >
auto MakeUnstructuredBoundaryFaceConnectivity(
   const BoundaryBuckets< Geometry > & buckets,
   std::integer_sequence< Integer, I... > )
{
   return std::make_tuple(
      UnstructuredBoundaryFaceConnectivity< Geometry, I >( buckets[ I ] )... );
}

template < typename Geometry >
auto MakeUnstructuredBoundaryFaceConnectivity(
   const BoundaryBuckets< Geometry > & buckets )
{
   return MakeUnstructuredBoundaryFaceConnectivity< Geometry >(
      buckets,
      std::make_integer_sequence< Integer, Geometry::num_faces >{} );
}

} // namespace unstructured_global_face_detail
} // namespace gendil
