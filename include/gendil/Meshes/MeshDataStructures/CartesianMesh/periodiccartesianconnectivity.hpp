// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"
#include "gendil/Meshes/Connectivities/computelinearindex.hpp"

namespace gendil
{

template < Integer Dim >
struct PeriodicCartesianConnectivity
{
   using geometry = HyperCube< Dim >;
   using orientation_type = IdentityOrientation< Dim >;
   using conformity_type = ConformingFaceMap< Dim >;
   using boundary_type = std::integral_constant< bool, false >;
   template < Integer FaceIndex, Integer NormalAxis = FaceIndex % Dim, int NormalSign = FaceIndex < Dim ? -1 : 1 >
   using face_info_type =
      ConformingCellFaceView <
         geometry,
         std::integral_constant< Integer, FaceIndex >,
         std::integral_constant< Integer, FaceIndex < Dim ? FaceIndex + Dim : FaceIndex - Dim >,
         orientation_type,
         CanonicalVector< Dim, NormalAxis, NormalSign >,
         CanonicalVector< Dim, NormalAxis, -NormalSign >,
         boundary_type
      >;

   std::array< GlobalIndex, Dim > sizes;

   template < typename ... Sizes >
   PeriodicCartesianConnectivity( const Sizes & ... sizes ):
      sizes( { (GlobalIndex)sizes... } )
   {}

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > ) const
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
      // to compute the neighbor.

      if ( sizes[Index] == 1 )
      {
         neighbor_index[Index] = 0;
      }
      else if ( neighbor_index[Index] == 0 )
      {
         if constexpr ( Sign == -1 )
         {
            neighbor_index[Index] = sizes[Index] - 1;
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
            neighbor_index[Index] = 0;
         }
         else
         {
            neighbor_index[Index]--;
         }
      }
      else
      {
         if constexpr ( Sign == 1 )
         {
            neighbor_index[Index] ++;
         }
         else
         {
            neighbor_index[Index] --;
         }
      }

      GlobalIndex neighbor_linear_index = ComputeLinearIndex( neighbor_index, sizes );

      return face_info_type<FaceIndex>{ { cell_index }, { neighbor_linear_index } };
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
struct PeriodicCartesianConnectivity< 1 >
{
   static constexpr Integer Dim = 1;
   using geometry = HyperCube< Dim >;
   using orientation_type = IdentityOrientation< Dim >;
   using conformity_type = ConformingFaceMap< Dim >;
   using boundary_type = std::integral_constant< bool, false >;
   template < Integer FaceIndex, Integer NormalAxis = FaceIndex % Dim, int NormalSign = FaceIndex < Dim ? -1 : 1 >
   using face_info_type =
      ConformingCellFaceView <
         geometry,
         std::integral_constant< Integer, FaceIndex >,
         std::integral_constant< Integer, FaceIndex < Dim ? FaceIndex + Dim : FaceIndex - Dim >,
         orientation_type,
         CanonicalVector< Dim, NormalAxis, NormalSign >,
         CanonicalVector< Dim, NormalAxis, -NormalSign >,
         boundary_type
      >;

   GlobalIndex size;

   PeriodicCartesianConnectivity( const Integer & size ):
      size( size )
   {}

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      static_assert(
         FaceIndex < 2,
         "FaceIndex out of bound."
      );

      GlobalIndex neighbor_index = cell_index;
      // !FIXME: This is magic and specific to HyperCube
      constexpr int Sign = FaceIndex < Dim ? -1 : 1;

      if ( size == 1 )
      {
         neighbor_index = 0;
      }
      else if ( neighbor_index == 0 )
      {
         if constexpr ( Sign == -1 )
         {
            neighbor_index = size - 1;
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
            neighbor_index = 0;
         }
         else
         {
            neighbor_index--;
         }
      }
      else
      {
         if constexpr ( Sign == 1 )
         {
            neighbor_index++;
         }
         else
         {
            neighbor_index--;
         }
      }

      return face_info_type<FaceIndex>{ { cell_index }, { neighbor_index } };
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