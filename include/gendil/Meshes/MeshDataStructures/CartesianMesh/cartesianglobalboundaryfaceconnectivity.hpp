// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/computelinearindex.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"

namespace gendil {

template < Integer Dim, Integer LocalFaceIndex >
struct CartesianBoundaryFaceConnectivity
{
   static_assert(
      LocalFaceIndex < 2 * Dim,
      "CartesianBoundaryFaceConnectivity: local face index out of bounds.");

   using geometry = HyperCube< Dim >;
   using orientation_type = IdentityOrientation< Dim >;

   static constexpr Integer local_face_index = LocalFaceIndex;
   static constexpr Integer neighbor_local_face_index =
      HyperCube< Dim >::GetOppositeFaceIndex( LocalFaceIndex );
   static constexpr Integer dim_index =
      HyperCube< Dim >::GetNormalDimensionIndex( LocalFaceIndex );
   static constexpr int sign =
      HyperCube< Dim >::GetNormalSign( LocalFaceIndex );

   using face_info_type =
      ConformingCellFaceView<
         geometry,
         std::integral_constant< Integer, local_face_index >,
         std::integral_constant< Integer, neighbor_local_face_index >,
         orientation_type,
         CanonicalVector< Dim, dim_index, sign >,
         CanonicalVector< Dim, dim_index, -sign >,
         bool
      >;

   std::array< GlobalIndex, Dim > sizes;
   GlobalIndex num_faces;

   GENDIL_HOST_DEVICE
   CartesianBoundaryFaceConnectivity(std::array< GlobalIndex, Dim > sizes_)
      : sizes(sizes_)
   {
      num_faces = 1;
      for (size_t i = 0; i < Dim; i++)
      {
         if (i != dim_index)
         {
            num_faces *= sizes[i];
         }
      }
   }

   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const
   {
      return num_faces;
   }

   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex& face_index) const
   {
      std::array< GlobalIndex, Dim > face_sizes = sizes;
      face_sizes[dim_index] = 1;
      auto face_indices = GetStructuredSubIndices(face_index, face_sizes);

      face_indices[dim_index] = sign > 0 ? sizes[dim_index] - 1 : 0;
      const GlobalIndex cell_linear_index =
         ComputeLinearIndex(face_indices, sizes);

      return face_info_type{
         { cell_linear_index, {}, {}, {}, {}, true },
         { GlobalIndex{}, {}, {}, {}, {}, true }
      };
   }
};

template < Integer Dim, Integer... I >
auto MakeCartesianBoundaryFaceConnectivity(
   const std::array< GlobalIndex, Dim >& sizes,
   std::index_sequence< I... >)
{
   return std::make_tuple(CartesianBoundaryFaceConnectivity< Dim, I >(sizes)...);
}

template < Integer Dim >
auto MakeCartesianBoundaryFaceConnectivity(
   const std::array< GlobalIndex, Dim >& sizes)
{
   return MakeCartesianBoundaryFaceConnectivity(
      sizes,
      std::make_index_sequence< 2 * Dim >{});
}

template < Integer Dim >
using CartesianBoundaryFaceConnectivityTuple =
   decltype(
      MakeCartesianBoundaryFaceConnectivity(
         std::array< GlobalIndex, Dim >{},
         std::make_index_sequence< 2 * Dim >{}));

} // namespace gendil
