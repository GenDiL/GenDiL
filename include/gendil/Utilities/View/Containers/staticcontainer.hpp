// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

template < typename T, size_t Size, typename IndexType = GlobalIndex >
struct StaticContainer
{
   using element_type = T;
   using index_type = IndexType;

   T data[ Size ];

   GENDIL_HOST_DEVICE
   T& operator[]( index_type const & index )
   {
      return data[ index ];
   }

   GENDIL_HOST_DEVICE
   const T& operator[]( index_type const & index ) const
   {
      return data[ index ];
   }
};

}
