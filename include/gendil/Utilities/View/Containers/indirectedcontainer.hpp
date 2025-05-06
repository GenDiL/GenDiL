// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include <iostream>

namespace gendil {

namespace detail {

   template < typename T >
   struct get_element_type
   {
      using type = typename T::element_type;
   };

   template < typename T >
   struct get_element_type< T* >
   {
      using type = T;
   };

   template < typename T >
   using get_element_type_t = typename get_element_type< T >::type;

}

template < typename Container, typename IndexType = GlobalIndex >
struct IndirectedContainer
{
   using element_type = detail::get_element_type_t< Container >;
   using index_type = IndexType;

   Container data;
   HostDevicePointer< const int > indirections;

   GENDIL_HOST_DEVICE
   element_type& operator[]( index_type const & index )
   {
      return data[ indirections[ index ] ];
   }

   GENDIL_HOST_DEVICE
   const element_type& operator[]( index_type const & index ) const
   {
      return data[ indirections[ index ] ];
   }
};

}
