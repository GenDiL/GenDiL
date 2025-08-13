// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/computelinearindex.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"

namespace gendil {

template < Integer Dim >
struct CartesianConnectivity
{
   using geometry = HyperCube< Dim >;
   using orientation_type = Permutation< Dim >;
   // Requires C++20
   // using orientation_type = std::integral_constant< Permutation<Dim>, MakeReferencePermutation< Dim >() >;
   using boundary_type = bool;

   std::array< GlobalIndex, Dim > sizes;

   template < typename ... Sizes >
   CartesianConnectivity( const Sizes & ... sizes ):
      sizes( { (GlobalIndex)sizes... } )
   {}

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto operator()( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      static_assert(
         FaceIndex < 2*Dim,
         "FaceIndex out of bound."
      );

      // !FIXME: This is magic and specific to HyperCube
      constexpr Integer Index = FaceIndex % Dim; // HyperCube< Dim >::GetNormalDimensionIndex( face_index ) ?
      constexpr int Sign = FaceIndex < Dim ? -1 : 1; // HyperCube< Dim >::GetNormalSign( face_index ) ?

      std::array< GlobalIndex, Dim > neighbor_index = GetStructuredSubIndices( cell_index, sizes );
      // TODO: we can forgo computing all of the indices (computing only
      // neighbor_index[Index] via GetStructuredSubIndex<Index>) and use strides
      // to compute the neighbor. (Might be better for GPU since it may use
      // fewer registers).

      bool boundary = false;

      if ( sizes[Index] == 1 )
      {
         neighbor_index[Index] = std::numeric_limits< GlobalIndex >::quiet_NaN();
         boundary = true;
      }
      else if ( neighbor_index[Index] == 0 )
      {
         if constexpr ( Sign == -1 )
         {
            neighbor_index[Index] = std::numeric_limits< GlobalIndex >::quiet_NaN();
            boundary = true;
         }
         else
         {
            neighbor_index[Index]++;
         }
      }
      else if ( neighbor_index[Index] == sizes[Index] - 1 )
      {
         if constexpr ( Sign == 1 )
         {
            neighbor_index[Index] = std::numeric_limits< GlobalIndex >::quiet_NaN();
            boundary = true;
         }
         else
         {
            neighbor_index[Index]--;
         }
      }
      else
      {
         if constexpr ( Sign == 1 )
            neighbor_index[Index]++;
         else
            neighbor_index[Index]--;
      }

      GlobalIndex neighbor_linear_index =
         boundary ?
            std::numeric_limits< GlobalIndex >::quiet_NaN() :
            ComputeLinearIndex( neighbor_index, sizes );

      using normal_type = CanonicalVector< Dim, Index, Sign >;
      using FaceInfo =
         FaceConnectivity<
            FaceIndex,
            geometry,
            orientation_type,
            boundary_type,
            normal_type
         >;
      return FaceInfo{ neighbor_linear_index, MakeReferencePermutation< Dim >(), boundary };
      // Requires C++20
      // return FaceInfo{ { neighbor_index, neighbor_linear_index } };
   }

   GENDIL_HOST_DEVICE
   GlobalIndex Size( GlobalIndex index ) const
   {
      return sizes[ index ];
   }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return Product( sizes );
   }
};

template < >
struct CartesianConnectivity< 1 >
{
   static constexpr Integer Dim = 1;
   using geometry = HyperCube< Dim >;
   using orientation_type = Permutation< Dim >;
   // Requires C++20
   // using orientation_type = std::integral_constant< Permutation<Dim>, MakeReferencePermutation< Dim >() >;
   using boundary_type = bool;

   GlobalIndex size;

   CartesianConnectivity( const Integer & size ):
      size( size )
   {}

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto operator()( GlobalIndex neighbor_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      static_assert(
         FaceIndex < 2,
         "FaceIndex out of bound."
      );

      // !FIXME: This is magic and specific to HyperCube
      constexpr Integer Index = FaceIndex % Dim;
      constexpr int Sign = FaceIndex < Dim ? -1 : 1;

      bool boundary = false;

      if ( size == 1 )
      {
         neighbor_index = std::numeric_limits< GlobalIndex >::quiet_NaN();
         boundary = true;
      }
      else if ( neighbor_index == 0 )
      {
         if constexpr ( Sign == -1 )
         {
            neighbor_index = std::numeric_limits< GlobalIndex >::quiet_NaN();
            boundary = true;
         }
         else
         {
            neighbor_index++;
         }
      }
      else if ( neighbor_index == size - 1 )
      {
         if constexpr ( Sign == 1 )
         {
            neighbor_index = std::numeric_limits< GlobalIndex >::quiet_NaN();
            boundary = true;
         }
         else
         {
            neighbor_index--;
         }
      }
      else
      {
         neighbor_index += Sign;
      }

      using normal_type = CanonicalVector< Dim, Index, Sign >;
      using FaceInfo =
         FaceConnectivity<
            FaceIndex,
            geometry,
            orientation_type,
            boundary_type,
            normal_type
         >;
      return FaceInfo{ neighbor_index, MakeReferencePermutation< Dim >(), boundary };
      // Requires C++20
      // return FaceInfo{ neighbor_index };
   }

   GENDIL_HOST_DEVICE
   GlobalIndex Size( GlobalIndex index ) const
   {
      return size;
   }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return size;
   }
};

}