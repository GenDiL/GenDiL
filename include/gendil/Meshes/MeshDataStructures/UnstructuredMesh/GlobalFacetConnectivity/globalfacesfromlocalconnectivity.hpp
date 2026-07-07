// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Restricted fallback for deriving global faces from trusted local
 * unstructured connectivity.
 *
 * This adapter derives global face families from trusted conforming local
 * connectivity. It performs only local access-safety checks: valid neighbor
 * indices and explicit v1 self-neighbor rejection. It does not validate
 * reciprocal half-face pairing or orientation composition.
 *
 * Multiple distinct facets between the same distinct cell pair are allowed and
 * emitted independently. This is not the direct MFEM import path; direct MFEM
 * global-face builders must not use this fallback.
 */

#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/LocalFacetConnectivity/unstructuredconformingconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/temporaryglobalfacematerialization.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

namespace gendil {

/**
 * @brief Restricted fallback for deriving global faces from trusted local
 * unstructured connectivity.
 *
 * This path is intended for trusted uniform conforming local connectivity and
 * is not used by the direct MFEM global-face importer.
 */
template < typename Geometry, Integer FaceIndex >
void AddUnstructuredInteriorFaceRecordsForFace(
   const UnstructuredConformingConnectivity< Geometry > & connectivity,
   const GlobalIndex num_cells,
   unstructured_global_face_detail::InteriorBuckets< Geometry > & buckets,
   std::integral_constant< Integer, FaceIndex > )
{
   for ( GlobalIndex cell = 0; cell < num_cells; ++cell )
   {
      const auto local_face_info =
         connectivity.GetLocalFaceInfo( cell, std::integral_constant< Integer, FaceIndex >{} );

      if ( IsBoundaryFace( local_face_info ) )
      {
         continue;
      }

      const GlobalIndex neighbor = local_face_info.PlusSide().GetCellIndex();
      GENDIL_VERIFY(
         neighbor < num_cells,
         "Unstructured global face connectivity found an invalid interior neighbor index.");
      GENDIL_VERIFY(
         neighbor != cell,
         "Unstructured global face connectivity found a self-neighbor interior face.");

      if ( cell < neighbor )
      {
         buckets[ FaceIndex ].push_back(
            { cell, neighbor, local_face_info.PlusSide().GetOrientation() } );
      }
   }
}

template < typename Geometry >
auto MakeUnstructuredInteriorFaceConnectivity(
   const UnstructuredConformingConnectivity< Geometry > & connectivity,
   const GlobalIndex num_cells )
{
   static_assert(
      Geometry::num_faces == 2 * Geometry::geometry_dim,
      "Unstructured global face connectivity currently supports hypercube geometries only.");

   unstructured_global_face_detail::InteriorBuckets< Geometry > buckets;

   ConstexprLoop< Geometry::num_faces >(
      [&] ( auto face )
      {
         AddUnstructuredInteriorFaceRecordsForFace(
            connectivity,
            num_cells,
            buckets,
            face );
      } );

   return unstructured_global_face_detail::MakeUnstructuredInteriorFaceConnectivity< Geometry >(
      buckets );
}

template < typename Geometry, Integer FaceIndex >
void AddUnstructuredBoundaryFaceRecordsForFace(
   const UnstructuredConformingConnectivity< Geometry > & connectivity,
   const GlobalIndex num_cells,
   unstructured_global_face_detail::BoundaryBuckets< Geometry > & buckets,
   std::integral_constant< Integer, FaceIndex > )
{
   for ( GlobalIndex cell = 0; cell < num_cells; ++cell )
   {
      const auto local_face_info =
         connectivity.GetLocalFaceInfo( cell, std::integral_constant< Integer, FaceIndex >{} );

      if ( !IsBoundaryFace( local_face_info ) )
      {
         continue;
      }

      buckets[ FaceIndex ].push_back( { cell } );
   }
}

template < typename Geometry >
auto MakeUnstructuredBoundaryFaceConnectivity(
   const UnstructuredConformingConnectivity< Geometry > & connectivity,
   const GlobalIndex num_cells )
{
   static_assert(
      Geometry::num_faces == 2 * Geometry::geometry_dim,
      "Unstructured global face connectivity currently supports hypercube geometries only.");

   unstructured_global_face_detail::BoundaryBuckets< Geometry > buckets;

   ConstexprLoop< Geometry::num_faces >(
      [&] ( auto face )
      {
         AddUnstructuredBoundaryFaceRecordsForFace(
            connectivity,
            num_cells,
            buckets,
            face );
      } );

   return unstructured_global_face_detail::MakeUnstructuredBoundaryFaceConnectivity< Geometry >(
      buckets );
}

template < typename Mesh >
auto MakeUnstructuredInteriorFaceConnectivity( const Mesh & mesh )
{
   return MakeUnstructuredInteriorFaceConnectivity(
      mesh.connectivity,
      mesh.GetNumberOfCells() );
}

template < typename Mesh >
auto MakeUnstructuredBoundaryFaceConnectivity( const Mesh & mesh )
{
   return MakeUnstructuredBoundaryFaceConnectivity(
      mesh.connectivity,
      mesh.GetNumberOfCells() );
}

} // namespace gendil
