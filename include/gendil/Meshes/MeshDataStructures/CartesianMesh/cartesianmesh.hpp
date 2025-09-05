// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// A 1d cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesian1dmesh.hpp"

// A 2d cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesian2dmesh.hpp"

// A 3d cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesian3dmesh.hpp"

// A face connectivity describing local face connectivity on a Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianlocalfaceconnectivity.hpp"

// A face connectivity describing local face connectivity on a periodic Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/periodiccartesianconnectivity.hpp"

// A face connectivity describing global face connectivity on a Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianglobalfaceconnectivity.hpp"

// An intermesh face connectivity between two conforming Cartesian meshes
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianintermeshfaceconnectivity.hpp"

// An intermesh face connectivity between two nonconforming Cartesian meshes
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/nonconformingcartesianintermeshfaceconnectivity.hpp"

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/computelinearindex.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/hypercubecell.hpp"


namespace gendil {

template < Integer Dim_ >
struct CartesianMesh
{
   static constexpr Integer Dim = Dim_;
   using geometry = HyperCube< Dim >;
   using cell_type = HyperCubeCell< Dim >;
   using orientation_type = Permutation< Dim >;
   // Requires C++20
   // using orientation_type = std::integral_constant< Permutation<Dim>, MakeReferencePermutation< Dim >() >;
   using boundary_type = bool;

   std::array< GlobalIndex, Dim > sizes;
   const std::array< Real, Dim > h;
   Point< Dim > mesh_origin;
   std::array< boundary_type, Dim > is_periodic_boundary;

   CartesianMesh(
      const std::array< GlobalIndex, Dim > & sizes,
      const std::array< Real, Dim > & h,
      const Point< Dim > & mesh_origin )
      : sizes( sizes ), h(h), mesh_origin( mesh_origin )
   {
      ConstexprLoop< Dim >( [&] ( auto i )
      {
         is_periodic_boundary[i] = false;
      });
   }

   CartesianMesh(
      const std::array< GlobalIndex, Dim > & sizes,
      const std::array< Real, Dim > & h,
      const Point< Dim > & mesh_origin,
      const bool periodic )
      : sizes( sizes ), h(h), mesh_origin( mesh_origin )
   {
      ConstexprLoop< Dim >( [&] ( auto i )
      {
         is_periodic_boundary[i] = periodic;
      });
   }

   CartesianMesh(
      const std::array< GlobalIndex, Dim > & sizes,
      const std::array< Real, Dim > & h,
      const Point< Dim > & mesh_origin,
      const std::array< int, Dim > & periodic_boundary )
      : sizes( sizes ), h(h), mesh_origin( mesh_origin )
   {
      ConstexprLoop< Dim >( [&] ( auto i )
      {
         is_periodic_boundary[i] = periodic_boundary[i];
      });
   }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return Product( sizes );
   }

   GENDIL_HOST_DEVICE
   GlobalIndex Size( GlobalIndex index ) const
   {
      return sizes[ index ];
   }

   GENDIL_HOST_DEVICE
   HyperCubeCell< Dim > GetCell( GlobalIndex element_index ) const
   {
      const auto index = GetStructuredSubIndices( element_index, sizes );

      Point< Dim > cell_origin;
      ConstexprLoop< Dim >( [&] ( auto i )
      {
         cell_origin[i] = mesh_origin[i] + h[i] * index[i];
      });
      return HyperCubeCell< Dim >( cell_origin, h );
   }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetFaceNeighborInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > face_index ) const
   {
      static_assert(
         FaceIndex < 2*Dim,
         "FaceIndex out of bound."
      );

      constexpr Integer Index = cell_type::GetNormalDimensionIndex( face_index );
      constexpr int Sign = cell_type::GetNormalSign( face_index );

      std::array< GlobalIndex, Dim > neighbor_index = GetStructuredSubIndices( cell_index, sizes );
      // TODO: we can forgo computing all of the indices (computing only
      // neighbor_index[Index] via GetStructuredSubIndex<Index>) and use strides
      // to compute the neighbor. (Might be better for GPU since it may use
      // fewer registers).

      bool boundary = false;

      if ( sizes[Index] == 1 )
      {
         if (is_periodic_boundary[Index])
         {
            neighbor_index[Index] = 0;
         }
         else
         {
            neighbor_index[Index] = std::numeric_limits< GlobalIndex >::max();
            boundary = true;
         }
      }
      else if ( neighbor_index[Index] == 0 )
      {
         if constexpr ( Sign == -1 )
         {
            if (is_periodic_boundary[Index])
            {
               neighbor_index[Index] = sizes[Index] - 1;
            }
            else
            {
               neighbor_index[Index] = std::numeric_limits< GlobalIndex >::max();
               boundary = true;
            }
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
            if (is_periodic_boundary[Index])
            {
               neighbor_index[Index] = 0;
            }
            else
            {
               neighbor_index[Index] = std::numeric_limits< GlobalIndex >::max();
               boundary = true;
            }
         }
         else
         {
            neighbor_index[Index]--;
         }
      }
      else
      {
         neighbor_index[Index] += Sign;
      }

      GlobalIndex neighbor_linear_index = boundary ? -1 : ComputeLinearIndex( neighbor_index, sizes );

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
};

}
