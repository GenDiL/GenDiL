// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"

namespace gendil {

template < Integer Dim >
struct ParallelCartesianMesh;

template < Integer Dim >
struct CartesianHalo
{

   static constexpr Integer num_neighbors = 2*Dim;
   std::array< int, num_neighbors > neighbors;
   GlobalIndex halo_size;
   std::array< GlobalIndex, num_neighbors > halo_offsets;
   std::array< GlobalIndex, num_neighbors > halo_sizes;
   // std::array< StridedLayout< Dim+1 > > halo_layouts; // TODO Use Fixed StridedLayout
#ifdef GENDIL_USE_MPI
   MPI_Comm communicator;
   int my_rank;
#endif

#ifdef GENDIL_USE_MPI
   CartesianHalo(
      const std::array< GlobalIndex, Dim > sizes,
      const std::array< int, Dim > & periodic_boundary,
      const std::array< int, Dim > & partition )
   {
      int size;
      MPI_Comm_size( MPI_COMM_WORLD, &size );
      int partition_size = Product( partition );
      Assert(
         partition_size <= size,
         "The partition requires more ranks than available." );
      int reorder = true;

      MPI_Cart_create( MPI_COMM_WORLD, Dim, partition.data(), periodic_boundary.data(), reorder, &communicator );

      MPI_Comm_rank( communicator, &my_rank );

      int my_coords[Dim];
      MPI_Cart_coords( communicator, my_rank, Dim, my_coords );

      ConstexprLoop< Dim >( [&] ( auto i )
      {
         MPI_Cart_shift( communicator, i, 1, &neighbors[i], &neighbors[Dim+i] );
      });

      halo_size = 0;
      for (size_t neighbor = 0; neighbor < num_neighbors; neighbor++)
      {
         halo_offsets[neighbor] = halo_size;
         if (neighbors[neighbor] != MPI_PROC_NULL)
         {
            const GlobalIndex dim = neighbor % Dim; // !FIXME Abstract behind function? Same problem as FaceIndex
            std::array< GlobalIndex, Dim > local_halo_sizes = sizes;
            local_halo_sizes[dim] = 1;
            const GlobalIndex local_halo_size = Product( local_halo_sizes );
            halo_size += local_halo_size;
            halo_sizes[neighbor] = local_halo_size;
         }
         else
         {
            halo_sizes[neighbor] = 0;
         }
      }
   }
#endif

   template < typename T, Integer ... Dims >
   struct CartesianInteriorHalo
   {
      using layout = decltype( MakeFixedFIFOStridedLayout<Dims...,0>() );
      // TODO use single pointer
      std::array< HostDevicePointer< T >, num_neighbors > data;
      std::array<  View< HostDevicePointer< T >, layout >, num_neighbors > halos;

      CartesianInteriorHalo( const std::array< GlobalIndex, Dim > & sizes, const std::array< int, num_neighbors > & neighbors )
      {
         // std::array<size_t,Dim> dofs_sizes = GetDofsSizes( typename FiniteElement::shape_functions{} );
         for (size_t neighbor = 0; neighbor < num_neighbors; neighbor++)
         {
            if (neighbors[neighbor] != MPI_PROC_NULL)
            {
               // !FIXME recomputing halo_sizes
               const GlobalIndex dim = neighbor % Dim;
               std::array< GlobalIndex, Dim > local_halo_sizes = sizes;
               local_halo_sizes[dim] = 1;
               const GlobalIndex halo_size = Product( local_halo_sizes ) * Product( Dims... );
               AllocateHostPointer( halo_size, data[neighbor] );
               halos[neighbor] = MakeView( data[neighbor], layout{} );
            }
            else
            {
               // data[neighbor] = nullptr;
               halos[neighbor] = MakeView( data[neighbor], layout{} );
            }
         }
      }

      template < typename FiniteElement, typename Restriction >
      CartesianInteriorHalo(
         const FiniteElementSpace< ParallelCartesianMesh< Dim >, FiniteElement, Restriction > & mesh )
         : CartesianInteriorHalo( mesh.sizes, mesh.halo.neighbors )
      {

      }

      T* operator[]( GlobalIndex halo_index ) const
      {
         return data[halo_index];
      }
   };

   template < typename T, typename Shape >
   struct interior_halo;

   template < typename T, Integer... Dims >
   struct interior_halo< T, std::index_sequence< Dims... > >
   {
      using type = CartesianInteriorHalo< T, Dims... >;
   };

   template < typename T, typename DofShape >
   using interior_halo_type = typename interior_halo< T, DofShape >::type;

   template < typename T, Integer... Dims >
   struct CartesianExteriorHalo
   {
      using layout = decltype( MakeFixedFIFOStridedLayout<Dims...,0>() ); // The zero is for the element_index
      HostDevicePointer< T > data;
      std::array< GlobalIndex, num_neighbors > halo_offsets;

      CartesianExteriorHalo(
         size_t halo_size,
         const std::array< GlobalIndex, Dim > & sizes,
         const std::array< int, num_neighbors > & neighbors,
         std::array< GlobalIndex, num_neighbors > mesh_halo_offsets )
      {
         for (size_t neighbor = 0; neighbor < num_neighbors; neighbor++)
         {
            halo_offsets[neighbor] = mesh_halo_offsets[neighbor] * Product( Dims... );
         }
         const GlobalIndex halo_data_size = halo_size * Product( Dims... );
         AllocateHostPointer( halo_data_size, data );
         AllocateDevicePointer( halo_data_size, data );
      }

      template < typename FiniteElement, typename Restriction >
      CartesianExteriorHalo(
         const FiniteElementSpace< ParallelCartesianMesh< Dim >, FiniteElement, Restriction > & mesh )
         : CartesianExteriorHalo( mesh.halo.halo_size, mesh.sizes, mesh.halo.neighbors, mesh.halo.halo_offsets )
      {}

      T* operator[]( GlobalIndex halo_index ) const
      {
         return data + halo_offsets[halo_index];
      }

      View< PointerContainer<T>, layout > GetView() const
      {
         return MakeView( PointerContainer<T>{ data }, layout{} );
      }
   };

   template < typename T, typename Shape >
   struct exterior_halo;

   template < typename T, Integer... Dims >
   struct exterior_halo< T, std::index_sequence< Dims... > >
   {
      using type = CartesianExteriorHalo< T, Dims... >;
   };

   template < typename T, typename Shape >
   using exterior_halo_type = typename exterior_halo< T, Shape >::type;
};

}
