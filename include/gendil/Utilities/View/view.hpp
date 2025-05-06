// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "Containers/containers.hpp"

namespace gendil {

// TODO: Generalize? < typename ElementType, typename Extents, typename Layout, typename Accessor >
template < typename Container, typename Layout >
struct View
{
   using element_type = typename Container::element_type;

   static constexpr size_t rank = Layout::rank;

   Container data;
   Layout layout;

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   auto & operator()( Indices... idx ) const
   {
      return data[ layout.Offset( idx... ) ];
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   auto & operator()( Indices... idx )
   {
      return data[ layout.Offset( idx... ) ];
   }
};

template < typename Container, typename Layout >
GENDIL_HOST_DEVICE
auto MakeView( const Container & data, const Layout & layout )
{
   return View< Container, Layout >{ data, layout };
}

template < typename T, typename Layout >
GENDIL_HOST_DEVICE
auto MakeView( T* const data, Layout layout )
{
   return MakeView( PointerContainer< T >{ data }, layout );
}

template < typename T, typename Layout >
GENDIL_HOST_DEVICE
auto MakeView( const HostDevicePointer< T > & data, const Layout & layout )
{
   return View< HostDevicePointer< T >, Layout >{ data, layout };
}

}