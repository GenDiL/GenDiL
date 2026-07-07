// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Unstructured global boundary connectivity for one static local-face
 * family.
 *
 * Boundary execution is one-sided. The minus side is the real boundary cell;
 * the plus side is a legacy dummy view required by GlobalFaceInfo shape and
 * must not be consumed as a valid cell.
 *
 * Construction materializes a snapshot in GenDiL-managed host/device storage.
 * Connectivity copies share that storage. No source-mesh pointer is retained.
 */

#include <vector>
#include <type_traits>

#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/MemoryManagement/garbagecollector.hpp"

namespace gendil {

/**
 * @brief Transitional execution payload for one boundary face.
 *
 * This remains publicly constructible while current builders materialize from
 * record vectors.
 */
struct UnstructuredBoundaryFaceRecord
{
   GlobalIndex cell = 0;
};

/**
 * @brief Unstructured boundary faces for one static local face family.
 *
 * One concrete connectivity type represents one statically typed execution
 * family. The minus side is the real boundary cell. The plus side is a legacy
 * dummy view used to satisfy GlobalFaceInfo shape requirements; current
 * boundary execution is one-sided and consumers must not read the dummy plus
 * cell.
 */
template < typename Geometry, Integer LocalFaceIndex >
struct UnstructuredBoundaryFaceConnectivity
{
   static_assert(
      LocalFaceIndex < Geometry::num_faces,
      "UnstructuredBoundaryFaceConnectivity: local face index out of bounds.");

   static constexpr Integer dim = Geometry::geometry_dim;
   static constexpr Integer local_face_index = LocalFaceIndex;
   static constexpr Integer neighbor_local_face_index =
      HyperCube< dim >::GetOppositeFaceIndex( LocalFaceIndex );
   static constexpr Integer dim_index =
      HyperCube< dim >::GetNormalDimensionIndex( LocalFaceIndex );
   static constexpr int sign =
      HyperCube< dim >::GetNormalSign( LocalFaceIndex );

   using geometry = Geometry;
   using record_type = UnstructuredBoundaryFaceRecord;
   using orientation_type = IdentityOrientation< dim >;
   using face_info_type =
      ConformingCellFaceView<
         geometry,
         std::integral_constant< Integer, local_face_index >,
         std::integral_constant< Integer, neighbor_local_face_index >,
         orientation_type,
         CanonicalVector< dim, dim_index, sign >,
         CanonicalVector< dim, dim_index, -sign >,
         bool
      >;

   HostDevicePointer< record_type > records;
   GlobalIndex num_faces = 0;

   UnstructuredBoundaryFaceConnectivity() = default;

   explicit UnstructuredBoundaryFaceConnectivity(
      const std::vector< record_type > & host_records )
      : num_faces( static_cast< GlobalIndex >( host_records.size() ) )
   {
      if ( num_faces == 0 )
      {
         return;
      }

      AllocateHostPointer( num_faces, records );
      AllocateDevicePointer( num_faces, records );
      GarbageCollector::Instance().RegisterHostDevicePtr( records );

      for ( GlobalIndex i = 0; i < num_faces; ++i )
      {
         records.host_pointer[ i ] = host_records[ i ];
      }
      ToDevice( num_faces, records );
   }

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return num_faces;
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo( const GlobalIndex & face_index ) const
   {
      const auto record = records[ face_index ];
      return face_info_type{
         { record.cell, {}, {}, {}, {}, true },
         { GlobalIndex{}, {}, {}, {}, {}, true }
      };
   }
};

} // namespace gendil
