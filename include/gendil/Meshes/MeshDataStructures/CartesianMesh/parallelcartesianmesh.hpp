// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/computelinearindex.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/hypercubecell.hpp"
#include "gendil/Utilities/Loop/loop.hpp"
#include "gendil/Utilities/View/view.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianhalo.hpp"

namespace gendil {

template < Integer Dim_ >
struct ParallelCartesianMesh
{
   static constexpr Integer Dim = Dim_;
   using geometry = HyperCube< Dim >;
   using cell_type = HyperCubeCell< Dim >;
   using orientation_type = Permutation< Dim >;
   // Requires C++20
   // using orientation_type = std::integral_constant< Permutation<Dim>, MakeReferencePermutation< Dim >() >;
   using boundary_type = FaceType;

   static constexpr bool has_halo = true;
   std::array< GlobalIndex, Dim > sizes;
   const std::array< Real, Dim > h;
   Point< Dim > local_mesh_origin;
   std::array< boundary_type, Dim > boundary;


#ifdef GENDIL_USE_MPI
   using halo_type = CartesianHalo< Dim >;
#else
   using halo_type = EmptyHalo< Dim >;
#endif
   halo_type halo;

   ParallelCartesianMesh(
      const std::array< GlobalIndex, Dim > & local_sizes,
      const std::array< Real, Dim > & h,
      const Point< Dim > & mesh_origin,
      const std::array< int, Dim > & periodic_boundary,
      const std::array< int, Dim > & partition )
      : sizes( local_sizes ), h(h), local_mesh_origin( mesh_origin ), halo( sizes, periodic_boundary, partition )
   {
#ifdef GENDIL_USE_MPI
      int my_coords[Dim];
      MPI_Cart_coords( halo.communicator, halo.my_rank, Dim, my_coords );
   
      ConstexprLoop< Dim >( [&] ( auto i )
      {
         local_mesh_origin[i] = mesh_origin[i] + h[i] * sizes[i] * my_coords[i];
      });
#else
      if( Product( partition ) > 1 )
      {
         std::cout << "Partition must have size 1 without MPI support." << std::endl;
         std::abort();
      }
      ConstexprLoop< Dim >( [&] ( auto i )
      {
         local_mesh_origin[i] = mesh_origin[i];
      });
#endif
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
         cell_origin[i] = local_mesh_origin[i] + h[i] * index[i];
      });
      return HyperCubeCell< Dim >( cell_origin, h );
   }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetFaceNeighborInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > ) const
   {
      static_assert(
         FaceIndex < 2*Dim,
         "FaceIndex out of bound."
      );

      constexpr Integer Index = cell_type::GetNormalDimensionIndex( FaceIndex );
      constexpr int Sign = cell_type::GetNormalSign( FaceIndex );

      std::array< GlobalIndex, Dim > neighbor_index = GetStructuredSubIndices( cell_index, sizes );
      // TODO: we can forgo computing all of the indices (computing only
      // neighbor_index[Index] via GetStructuredSubIndex<Index>) and use strides
      // to compute the neighbor. (Might be better for GPU since it may use
      // fewer registers).

      FaceType face_type = Interior;
      std::array< GlobalIndex, Dim > local_sizes = sizes;
      GlobalIndex neighbor_linear_index = std::numeric_limits< GlobalIndex >::max();

      if ( sizes[Index] == 1 )
      {
         if ( halo.neighbors[FaceIndex] == MPI_PROC_NULL ) // True boundary
         {
            neighbor_index[Index] = std::numeric_limits< GlobalIndex >::max();
            face_type = Boundary;
            neighbor_linear_index = std::numeric_limits< GlobalIndex >::max();
         }
         else if ( halo.neighbors[FaceIndex] == halo.my_rank ) // local neighbor
         {
            neighbor_index[Index] = 0;
            face_type = Interior;
            neighbor_linear_index = ComputeLinearIndex( neighbor_index, local_sizes );
         }
         else // distributed neighbor
         {
            neighbor_index[Index] = 0;
            local_sizes[Index] = 1;
            face_type = Distributed;
            neighbor_linear_index = halo.halo_offsets[FaceIndex] + ComputeLinearIndex( neighbor_index, local_sizes );
         }
      }
      else if ( neighbor_index[Index] == 0 )
      {
         if constexpr ( Sign == -1 )
         {
            if ( halo.neighbors[FaceIndex] == MPI_PROC_NULL ) // True boundary
            {
               neighbor_index[Index] = std::numeric_limits< GlobalIndex >::max();
               face_type = Boundary;
               neighbor_linear_index = std::numeric_limits< GlobalIndex >::max();
            }
            else if ( halo.neighbors[FaceIndex] == halo.my_rank ) // local neighbor ( periodic mesh )
            {
               neighbor_index[Index] = sizes[Index] - 1;
               face_type = Interior;
               neighbor_linear_index = ComputeLinearIndex( neighbor_index, local_sizes );
            }
            else // distributed neighbor
            {
               neighbor_index[Index] = 0; // zero because it's the index in the halo
               local_sizes[Index] = 1;
               face_type = Distributed;
               neighbor_linear_index = halo.halo_offsets[FaceIndex] + ComputeLinearIndex( neighbor_index, local_sizes );
            }
         }
         else // local neighbor ( "normal" case )
         {
            neighbor_index[Index]++;
            face_type = Interior;
            neighbor_linear_index = ComputeLinearIndex( neighbor_index, local_sizes );
         }
      }
      else if ( neighbor_index[Index] == sizes[Index] - 1 )
      {
         if constexpr ( Sign == 1 )
         {
            if ( halo.neighbors[FaceIndex] == MPI_PROC_NULL ) // True boundary
            {
               neighbor_index[Index] = std::numeric_limits< GlobalIndex >::max();
               face_type = Boundary;
               neighbor_linear_index = std::numeric_limits< GlobalIndex >::max();
            }
            else if ( halo.neighbors[FaceIndex] == halo.my_rank ) // local neighbor ( periodic mesh )
            {
               neighbor_index[Index] = 0;
               face_type = Interior;
               neighbor_linear_index = ComputeLinearIndex( neighbor_index, local_sizes );
            }
            else // distributed neighbor
            {
               neighbor_index[Index] = 0; // zero because it's the index in the halo
               local_sizes[Index] = 1;
               face_type = Distributed;
               neighbor_linear_index = halo.halo_offsets[FaceIndex] + ComputeLinearIndex( neighbor_index, local_sizes );
            }
         }
         else // local neighbor ( "normal" case )
         {
            neighbor_index[Index]--;
            face_type = Interior;
            neighbor_linear_index = ComputeLinearIndex( neighbor_index, local_sizes );
         }
      }
      else // local neighbor ( "normal" case )
      {
         neighbor_index[Index] += Sign;
         face_type = Interior;
         neighbor_linear_index = ComputeLinearIndex( neighbor_index, local_sizes );
      }

      // std::cout << "Access Face " << FaceIndex << ": e=" << neighbor_linear_index << " b=" << face_type << std::endl;
      using normal_type = CanonicalVector< Dim, Index, Sign >;
      using FaceInfo =
         FaceConnectivity<
            FaceIndex,
            geometry,
            orientation_type,
            boundary_type,
            normal_type
         >;
      return FaceInfo{ neighbor_linear_index, MakeReferencePermutation< Dim >(), face_type };
      // Requires C++20
      // return FaceInfo{ { neighbor_index, neighbor_linear_index } };
   }
};

// Expected semantic: [&]( auto halo_index, GlobalIndex halo_cell_index, GlobalIndex cell_index )
template < Integer Dim >
void ForEachInteriorHaloCell(
   const ParallelCartesianMesh< Dim > & mesh,
   std::function<void( GlobalIndex, GlobalIndex, GlobalIndex )> && body )
{
#ifdef GENDIL_USE_MPI
   Loop< ParallelCartesianMesh< Dim >::halo_type::num_neighbors >( [&] ( auto neighbor_index )
   {
      const int neighbor_rank = mesh.halo.neighbors[ neighbor_index ];
      if ( neighbor_rank != MPI_PROC_NULL )
      {
         const Integer dim_index = HyperCubeCell< Dim >::GetNormalDimensionIndex( neighbor_index );
         const int sign = HyperCubeCell< Dim >::GetNormalSign( neighbor_index );
         std::array< GlobalIndex, Dim > halo_sizes = mesh.sizes;
         halo_sizes[dim_index] = 1;
         // Pack the data
         // TODO Make this GPU enabled
         DynamicLoop( halo_sizes, [=]( auto ... halo_indices )
         {
            std::array< GlobalIndex, Dim > halo_index = { halo_indices... };
            std::array< GlobalIndex, Dim > cell_index = halo_index;
            cell_index[dim_index] = sign == -1 ? 0 : mesh.sizes[dim_index] - 1;
            // This might be too expensive...
            GlobalIndex linear_halo_index = ComputeLinearIndex( halo_index, halo_sizes );
            GlobalIndex linear_cell_index = ComputeLinearIndex( cell_index, mesh.sizes );

            body( neighbor_index, linear_halo_index, linear_cell_index );
         });
      }
   });
#endif
}

}
