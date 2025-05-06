// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <ostream>
#include "gendil/Utilities/types.hpp"

namespace gendil {

/**
 * @brief Simple structure representing a tensor index.
 * 
 * @tparam Dim The dimension of the index.
 */
template < Integer Dim >
struct TensorIndex
{
   GlobalIndex indices[Dim];

   GENDIL_HOST_DEVICE
   constexpr TensorIndex()
   {
      for (GlobalIndex i = 0; i < Dim; i++)
      {
         indices[ i ] = 0;
      }
   }

   constexpr TensorIndex( const TensorIndex & other ) = default;

   template < typename... Idx >
   GENDIL_HOST_DEVICE
   constexpr TensorIndex( GlobalIndex first_index, Idx... rest_indices ) :
      indices{ first_index, rest_indices... }
   { }

   GENDIL_HOST_DEVICE
   constexpr GlobalIndex operator[]( GlobalIndex index ) const
   {
      return indices[ index ];
   }

   GENDIL_HOST_DEVICE
   constexpr GlobalIndex & operator[]( GlobalIndex index )
   {
      return indices[ index ];
   }

   // TODO: Make a function instead?
   template < Integer Begin, Integer End, size_t... Is >
   GENDIL_HOST_DEVICE
   constexpr auto Sub( std::index_sequence<Is...> ) const
   {
      constexpr Integer size = End - Begin;
      return TensorIndex< size >( indices[Begin+Is]... );
   }

   template < Integer Begin, Integer End >
   GENDIL_HOST_DEVICE
   constexpr auto Sub() const
   {
      static_assert(
         Begin >= 0,
         "Begin must be positive."
      );
      static_assert(
         End <= Dim,
         "End must be smaller than the Dim."
      );
      constexpr Integer size = End - Begin;
      return Sub<Begin,End>( std::make_index_sequence< size >{} );
   }
};

template < size_t Dim >
std::ostream& operator<<(std::ostream& os, TensorIndex<Dim> const & index) {
   os << "( ";
   for (size_t i = 0; i < Dim; i++)
   {
      os << index[ i ] << " ";
   }
   os << ")";
   return os;
}

template < size_t... Dims, size_t Dim >
GENDIL_HOST_DEVICE
auto GetSubIndex( const TensorIndex< Dim > & input )
{
   constexpr size_t new_dim = sizeof...(Dims);
   return TensorIndex< new_dim >( input[ Dims ]... );
}

template < typename Input, size_t Dim, size_t... Is >
GENDIL_HOST_DEVICE
auto Apply( const Input & input, const TensorIndex< Dim > & index, std::index_sequence< Is... > )
{
   return input( index[ Is ]... );
}

template < typename Input, size_t Dim >
GENDIL_HOST_DEVICE
auto Apply( const Input & input, const TensorIndex< Dim > & index )
{
   return Apply( input, index, std::make_index_sequence< Dim >{} );
}

}
