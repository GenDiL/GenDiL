// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Local/cell-neighbor facet connectivity for conforming unstructured
 * meshes.
 */

#include <vector>
#include "gendil/Meshes/Connectivities/conformingcellconnectivity.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/MemoryManagement/garbagecollector.hpp"

namespace gendil {

template < typename Geometry >
struct UnstructuredConformingConnectivity
{
   static constexpr Integer dim = Geometry::geometry_dim;
   using geometry = Geometry;
   using orientation_type = Permutation< dim >;
   using boundary_type = bool;
   using conformity_type = ConformingFaceMap<dim>;
   template <
      Integer FaceIndex,
      Integer NormalAxis = HyperCube< dim >::GetNormalDimensionIndex( FaceIndex ),
      int NormalSign = HyperCube< dim >::GetNormalSign( FaceIndex ) >
   using face_info_type =
      ConformingCellFaceView <
         geometry,
         std::integral_constant< Integer, FaceIndex >,
         std::integral_constant< Integer, HyperCube< dim >::GetOppositeFaceIndex( FaceIndex ) >,
         orientation_type,
         CanonicalVector< dim, NormalAxis, NormalSign >,
         CanonicalVector< dim, NormalAxis, -NormalSign >,
         boundary_type
      >;

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
   auto GetLocalFaceInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      static_assert(
         FaceIndex < geometry::num_faces,
         "FaceIndex out of bound."
      );

      auto face_info = element_connectivities[ cell_index ].faces[ FaceIndex ];

      return face_info_type< FaceIndex >{
         { cell_index, {}, {}, {}, {}, face_info.boundary },
         { face_info.cell_index, {}, face_info.orientation, {}, {}, face_info.boundary }
      };
   }
};

}
