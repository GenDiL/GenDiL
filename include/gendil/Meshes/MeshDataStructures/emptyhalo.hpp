// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/repeat.hpp"

namespace gendil {

template < Integer Dim >
struct EmptyHalo
{
   GlobalIndex* halo_sizes;
   int* neighbors;
   static constexpr Integer num_neighbors = 0;
   #ifdef GENDIL_USE_MPI
   MPI_Comm communicator;
   #endif
   using layout = decltype( MakeFixedFIFOStridedLayout( repeat_t< 0, Dim+1 >{} ) );

   template < typename T, typename Shape >
   struct interior_halo_type
   {
      std::array< HostDevicePointer< T >, num_neighbors > data;

      template < typename Mesh >
      interior_halo_type( const Mesh & mesh ) { }

      T* operator[]( GlobalIndex halo_index ) const
      {
         return nullptr;
      }

      View< PointerContainer<T>, layout > GetView() const
      {
         return MakeView( PointerContainer<T>{ nullptr }, layout{} );
      }

   };

   template < typename T, typename Shape >
   using exterior_halo_type = interior_halo_type< T, Shape >;

};

}
