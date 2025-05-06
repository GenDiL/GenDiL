// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <ostream>

#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

/**
 * @brief Represent a permutation.
 * 
 * @tparam Rank 
 */
// TODO: make constexpr compatible
template < size_t Rank >
struct Permutation
{
   std::array< LocalIndex, Rank > dimension_indices;

   GENDIL_HOST_DEVICE
   const LocalIndex & operator()( size_t i ) const
   {
      return dimension_indices[ i ];
   }

   GENDIL_HOST_DEVICE
   LocalIndex & operator()( size_t i )
   {
      return dimension_indices[ i ];
   }
};

template < size_t Offset, size_t Size, size_t Rank >
GENDIL_HOST_DEVICE
void Set( Permutation< Rank > & permutation, Permutation< Size > const & sub_permutation )
{
   ConstexprLoop< Size >(
      [&](auto i) {
         LocalIndex dim_index = sub_permutation( i );
         permutation( Offset + i ) = (dim_index > 0) ? (Offset + dim_index) : - ( Offset + dim_index );
      }
   );
}

// Requires C++20
// template < size_t Offset, size_t Size, size_t Rank, Permutation< Size > sub_permutation >
// void Set( Permutation< Rank > & permutation, std::integral_constant< Permutation< Size >, sub_permutation > )
// {
//    Set< Offset >( permutation, sub_permutation, std::make_index_sequence<Size>{} );
// }

template < size_t Rank, size_t... Is >
constexpr auto MakeReferencePermutation( std::index_sequence< Is... > )
{
   return Permutation< Rank >( { (Is+1)... } );
}

template < size_t Rank >
constexpr auto MakeReferencePermutation()
{
   return MakeReferencePermutation< Rank >( std::make_index_sequence< Rank >{} );
}

template < size_t Rank >
std::ostream& operator<<(std::ostream& os, Permutation<Rank> const & orientation) {
   for (size_t i = 0; i < Rank; i++)
   {
      os << orientation( i ) << " ";
   }
   os << std::endl;
   return os;
}

}
