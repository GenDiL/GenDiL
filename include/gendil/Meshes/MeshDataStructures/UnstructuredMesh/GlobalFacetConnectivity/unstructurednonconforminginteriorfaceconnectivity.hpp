// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Nonconforming unstructured global interior connectivity for one
 * coarse canonical-minus face family.
 *
 * The minus side is the coarse/master cell and carries a
 * NonconformingHyperCubeFaceMap. The static plus face is the opposite
 * canonical face; the actual fine native face is recovered from the runtime
 * plus orientation.
 *
 * This header represents the backend-neutral face-info shape only.
 * Nonconforming MFEM construction, specialized execution, generic execution,
 * and map-aware geometry are separate capabilities.
 */

#include <type_traits>
#include <vector>

#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/MemoryManagement/garbagecollector.hpp"

namespace gendil {

template < Integer Dim >
/**
 * @brief Transitional execution payload for one nonconforming leaf face.
 *
 * This remains publicly constructible while current builders materialize from
 * record vectors.
 */
struct UnstructuredNonconformingInteriorFaceRecord
{
   GlobalIndex minus_cell = 0;
   GlobalIndex plus_cell = 0;
   Permutation< Dim > plus_orientation = MakeReferencePermutation< Dim >();
   NonconformingHyperCubeFaceMap< Dim > minus_nonconforming_map{};
};

/**
 * @brief Nonconforming unstructured interior leaf faces for one coarse static
 * local face family.
 *
 * One concrete connectivity type represents one statically typed execution
 * family. The minus side is the coarse/master cell and carries a
 * NonconformingHyperCubeFaceMap; the plus side is the fine/leaf cell on the
 * static opposite face and carries runtime plus orientation.
 *
 * Nonconforming representation, construction, specialized advection execution,
 * generic execution, and map-aware geometry are separate capabilities.
 */
template < typename Geometry, Integer LocalFaceIndex >
struct UnstructuredNonconformingInteriorFaceConnectivity
{
   static_assert(
      is_hypercube_geometry< Geometry >::value,
      "UnstructuredNonconformingInteriorFaceConnectivity currently supports hypercube geometries only.");
   static_assert(
      LocalFaceIndex < Geometry::num_faces,
      "UnstructuredNonconformingInteriorFaceConnectivity: local face index out of bounds.");

   static constexpr Integer dim = Geometry::geometry_dim;
   static constexpr Integer local_face_index = LocalFaceIndex;
   static constexpr Integer neighbor_local_face_index =
      HyperCube< dim >::GetOppositeFaceIndex( LocalFaceIndex );
   static constexpr Integer dim_index =
      HyperCube< dim >::GetNormalDimensionIndex( LocalFaceIndex );
   static constexpr int sign =
      HyperCube< dim >::GetNormalSign( LocalFaceIndex );

   using geometry = Geometry;
   using record_type = UnstructuredNonconformingInteriorFaceRecord< dim >;
   using orientation_type = Permutation< dim >;
   using minus_conformity_type = NonconformingHyperCubeFaceMap< dim >;
   using plus_conformity_type = ConformingFaceMap< dim >;
   static_assert(
      std::is_trivially_copyable_v< record_type >,
      "Unstructured nonconforming face records must be trivially copyable for device execution.");

   using minus_view_type =
      FaceView<
         std::integral_constant< Integer, local_face_index >,
         geometry,
         IdentityOrientation< dim >,
         CanonicalVector< dim, dim_index, sign >,
         minus_conformity_type,
         std::bool_constant< false >
      >;
   using plus_view_type =
      FaceView<
         std::integral_constant< Integer, neighbor_local_face_index >,
         geometry,
         orientation_type,
         CanonicalVector< dim, dim_index, -sign >,
         plus_conformity_type,
         std::bool_constant< false >
      >;
   using face_info_type = GlobalFaceInfo< minus_view_type, plus_view_type >;

   HostDevicePointer< record_type > records;
   GlobalIndex num_faces = 0;

   UnstructuredNonconformingInteriorFaceConnectivity() = default;

   explicit UnstructuredNonconformingInteriorFaceConnectivity(
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

   // Construction materializes a snapshot in GenDiL-managed host/device
   // storage. Connectivity copies share that storage.
   UnstructuredNonconformingInteriorFaceConnectivity(
      const UnstructuredNonconformingInteriorFaceConnectivity & ) = default;

   UnstructuredNonconformingInteriorFaceConnectivity & operator=(
      const UnstructuredNonconformingInteriorFaceConnectivity & other )
   {
      records.host_pointer = other.records.host_pointer;
#ifdef GENDIL_USE_DEVICE
      records.device_pointer = other.records.device_pointer;
#endif
      num_faces = other.num_faces;
      return *this;
   }

   UnstructuredNonconformingInteriorFaceConnectivity(
      UnstructuredNonconformingInteriorFaceConnectivity && ) noexcept = default;

   UnstructuredNonconformingInteriorFaceConnectivity & operator=(
      UnstructuredNonconformingInteriorFaceConnectivity && ) noexcept = default;

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return num_faces;
   }

   GENDIL_HOST_DEVICE
   static minus_view_type MakeMinusView(
      GlobalIndex cell,
      const minus_conformity_type & map )
   {
      return minus_view_type{
         .cell_index = cell,
         .local_face_index = {},
         .orientation = {},
         .normal = {},
         .conformity = map,
         .boundary = {}
      };
   }

   GENDIL_HOST_DEVICE
   static plus_view_type MakePlusView(
      GlobalIndex cell,
      const orientation_type & orientation )
   {
      return plus_view_type{
         .cell_index = cell,
         .local_face_index = {},
         .orientation = orientation,
         .normal = {},
         .conformity = {},
         .boundary = {}
      };
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo( const GlobalIndex & face_index ) const
   {
      const auto record = records[ face_index ];
      return face_info_type{
         MakeMinusView( record.minus_cell, record.minus_nonconforming_map ),
         MakePlusView( record.plus_cell, record.plus_orientation )
      };
   }
};

} // namespace gendil
