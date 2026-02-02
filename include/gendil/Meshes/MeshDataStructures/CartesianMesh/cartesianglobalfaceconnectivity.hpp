// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/computelinearindex.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"

namespace gendil {

template < Integer Dim, Integer LocalFaceIndex >
struct CartesianInteriorLocalFaceConnectivity
{
   using geometry = HyperCube< Dim >;
   using orientation_type = IdentityOrientation< Dim >;
   static constexpr Integer local_face_index = LocalFaceIndex;
   static constexpr Integer neighbor_local_face_index = LocalFaceIndex < Dim ? LocalFaceIndex + Dim : LocalFaceIndex - Dim;
   static constexpr Integer dim_index = LocalFaceIndex % Dim;
   static constexpr int sign = LocalFaceIndex < Dim ? -1 : 1;
   using normal_type = CanonicalVector< Dim, dim_index, sign >;
   using face_info_type =
      ConformingCellFaceView<
         geometry,
         std::integral_constant< Integer, LocalFaceIndex >,
         std::integral_constant< Integer, neighbor_local_face_index >,
         orientation_type,
         CanonicalVector< Dim, dim_index, +sign >,
         CanonicalVector< Dim, dim_index, -sign >
      >;

   std::array< GlobalIndex, Dim > sizes;
   GlobalIndex num_faces;

   GENDIL_HOST_DEVICE
   CartesianInteriorLocalFaceConnectivity( std::array< GlobalIndex, Dim > sizes_ ):
      sizes( sizes_ )
   {
      num_faces = 1;
      for (size_t i = 0; i < Dim; i++)
      {
         if ( i == dim_index )
         {
            num_faces *= sizes[i]-1;
         }
         else
         {
            num_faces *= sizes[i];
         }
      }
   }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfFaces() const
   {
      return num_faces;
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo( const GlobalIndex & face_index ) const
   {
      std::array< GlobalIndex, Dim > face_sizes = sizes;
      face_sizes[ dim_index ]--;
      std::array< GlobalIndex, Dim > face_indices = GetStructuredSubIndices( face_index, face_sizes );

      if constexpr ( sign == 1)
      {
         GlobalIndex minus_linear_index = ComputeLinearIndex( face_indices, sizes );
         face_indices[ dim_index ] ++;
         GlobalIndex plus_linear_index = ComputeLinearIndex( face_indices, sizes );
         return face_info_type{ { minus_linear_index }, { plus_linear_index } };
      }
      else
      {
         GlobalIndex plus_linear_index = ComputeLinearIndex( face_indices, sizes );
         face_indices[ dim_index ] ++;
         GlobalIndex minus_linear_index = ComputeLinearIndex( face_indices, sizes );
         return face_info_type{ { minus_linear_index }, { plus_linear_index } };
      }
   }
};

template < Integer Dim, Integer... I >
auto make_cartesian_interior_face_connectivity( const std::array< GlobalIndex, Dim > & sizes, std::index_sequence< I...  >)
{
   return std::make_tuple( CartesianInteriorLocalFaceConnectivity< Dim, I >( sizes )... );
}

template < Integer Dim >
auto make_cartesian_interior_face_connectivity( const std::array< GlobalIndex, Dim > & sizes )
{
   return make_cartesian_interior_face_connectivity( sizes, std::make_index_sequence< Dim >{} );
}

template < Integer Dim >
using CartesianInteriorFaceConnectivity =
   decltype(
      make_cartesian_interior_face_connectivity(
         std::array< GlobalIndex, Dim >{},
         std::make_index_sequence< Dim >{} ) );

} // namespace gendil