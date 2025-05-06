// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include "conformingcellconnectivity.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Utilities/MemoryManagement/garbagecollector.hpp"

namespace gendil {

template < typename Geometry >
struct UnstructuredConformingConnectivity
{
   static constexpr Integer dim = Geometry::geometry_dim;
   using geometry = Geometry;
   using orientation_type = Permutation< dim >;
   using boundary_type = bool;

   HostDevicePointer< ConformingCellConnectivity< Geometry > > element_connectivities;

   UnstructuredConformingConnectivity( const GlobalIndex & num_connectivities )
   {
      AllocateHostPointer( num_connectivities, element_connectivities );
      AllocateDevicePointer( num_connectivities, element_connectivities );
      GarbageCollector::Instance().RegisterHostDevicePtr( element_connectivities );
   }

   GENDIL_HOST_DEVICE
   const auto & operator[]( int i ) const
   {
      return element_connectivities[ i ];
   }

   GENDIL_HOST_DEVICE
   auto & operator[]( int i )
   {
      return element_connectivities[ i ];
   }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto operator()( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      static_assert(
         FaceIndex < geometry::num_faces,
         "FaceIndex out of bound."
      );

      auto face_info = element_connectivities[ cell_index ].faces[ FaceIndex ];

      // !FIXME: This is magic and specific to HyperCube
      constexpr Integer Index = FaceIndex % dim;
      constexpr int Sign = FaceIndex < dim ? -1 : 1;
      using normal_type = CanonicalVector< dim, Index, Sign >;
      using FaceInfo =
         FaceConnectivity<
            FaceIndex,
            geometry,
            orientation_type,
            boundary_type,
            normal_type
         >;
      return FaceInfo{ face_info.neighbor_index, face_info.orientation, face_info.boundary };
   }
};

}