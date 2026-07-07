// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Conforming unstructured global interior connectivity for one static
 * canonical-minus face family.
 *
 * The static plus face is the opposite canonical face. The actual native plus
 * face is recovered from the runtime plus orientation.
 *
 * Construction materializes a snapshot in GenDiL-managed host/device storage.
 * Connectivity copies share that storage. No source-mesh pointer is retained.
 * The public record payload is part of the current vector-materialization API.
 */

#include <vector>
#include <type_traits>

#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/MemoryManagement/garbagecollector.hpp"

namespace gendil {

template < Integer Dim >
/**
 * @brief Transitional execution payload for one conforming interior face.
 *
 * This remains publicly constructible while current builders materialize from
 * record vectors.
 */
struct UnstructuredInteriorFaceRecord
{
   GlobalIndex minus_cell = 0;
   GlobalIndex plus_cell = 0;
   Permutation< Dim > plus_orientation = MakeReferencePermutation< Dim >();
};

/**
 * @brief Conforming unstructured global interior faces for one static local
 * face family.
 *
 * One concrete connectivity type represents one statically typed execution
 * family, not an entire heterogeneous topology. Construction materializes a
 * snapshot in GenDiL-managed host/device storage. Connectivity copies share
 * that storage. No source-mesh pointer is retained.
 */
template < typename Geometry, Integer LocalFaceIndex >
struct UnstructuredInteriorFaceConnectivity
{
   static_assert(
      LocalFaceIndex < Geometry::num_faces,
      "UnstructuredInteriorFaceConnectivity: local face index out of bounds.");

   static constexpr Integer dim = Geometry::geometry_dim;
   static constexpr Integer local_face_index = LocalFaceIndex;
   static constexpr Integer neighbor_local_face_index =
      HyperCube< dim >::GetOppositeFaceIndex( LocalFaceIndex );
   static constexpr Integer dim_index =
      HyperCube< dim >::GetNormalDimensionIndex( LocalFaceIndex );
   static constexpr int sign =
      HyperCube< dim >::GetNormalSign( LocalFaceIndex );

   using geometry = Geometry;
   using record_type = UnstructuredInteriorFaceRecord< dim >;
   using orientation_type = Permutation< dim >;
   using face_info_type =
      ConformingCellFaceView<
         geometry,
         std::integral_constant< Integer, local_face_index >,
         std::integral_constant< Integer, neighbor_local_face_index >,
         orientation_type,
         CanonicalVector< dim, dim_index, sign >,
         CanonicalVector< dim, dim_index, -sign >
      >;

   HostDevicePointer< record_type > records;
   GlobalIndex num_faces = 0;

   UnstructuredInteriorFaceConnectivity() = default;

   explicit UnstructuredInteriorFaceConnectivity(
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
         { record.minus_cell },
         { record.plus_cell, {}, record.plus_orientation }
      };
   }
};

} // namespace gendil
