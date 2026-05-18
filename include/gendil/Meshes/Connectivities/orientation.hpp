// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <ostream>

#include "gendil/prelude.hpp"
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

template < size_t Rank >
GENDIL_HOST_DEVICE
bool operator==( const Permutation< Rank > & a,
                 const Permutation< Rank > & b )
{
   for ( size_t i = 0; i < Rank; ++i )
   {
      if ( a( i ) != b( i ) )
      {
         return false;
      }
   }
   return true;
}

template < size_t Rank >
GENDIL_HOST_DEVICE
bool operator!=( const Permutation< Rank > & a,
                 const Permutation< Rank > & b )
{
   return !( a == b );
}

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

template < size_t SubDim, size_t Dim >
GENDIL_HOST_DEVICE
auto GetSubPermutation( const Permutation< Dim > & orientation, size_t offset )
{
   Permutation< SubDim > sub_permutation{};
   for (size_t i = 0; i < SubDim; ++i)
   {
      sub_permutation( i ) = orientation( offset + i ) > 0
         ? orientation( offset + i ) - static_cast<LocalIndex>( offset )
         : orientation( offset + i ) + static_cast<LocalIndex>( offset );
   }

   return sub_permutation;
}

template <Integer Dim>
GENDIL_HOST_DEVICE
std::array<Integer, Dim> ReferenceToNativeIndex(
   const std::array<Integer, Dim>& k_ref,
   const std::array<size_t, Dim>& sizes,
   const Permutation<Dim>& orientation)
{
   std::array<Integer, Dim> k_native{};

   for (size_t j = 0; j < Dim; ++j)
   {
      const int o = orientation(j);
      const size_t p = static_cast<size_t>(o > 0 ? o - 1 : -o - 1);

      k_native[j] = (o > 0)
         ? k_ref[p]
         : static_cast<Integer>(sizes[j] - 1 - k_ref[p]);
   }

   return k_native;
}

template <Integer Dim, size_t... Is >
GENDIL_HOST_DEVICE
std::array<Integer, Dim> ReferenceToNativeIndex(
   const std::array<Integer, Dim>& k_ref,
   const std::index_sequence<Is...>& sizes,
   const Permutation<Dim>& orientation)
{
   return ReferenceToNativeIndex( k_ref, std::array<size_t, Dim>{ Is... }, orientation );
}

template <Integer Dim>
GENDIL_HOST_DEVICE
std::array<Integer, Dim> NativeToReferenceIndex(
   const std::array<Integer, Dim>& k_native,
   const std::array<size_t, Dim>& sizes,
   const Permutation<Dim>& orientation)
{
   std::array<Integer, Dim> k_ref{};

   for (size_t j = 0; j < Dim; ++j)
   {
      const int o = orientation(j);
      const size_t p = static_cast<size_t>(o > 0 ? o - 1 : -o - 1);

      k_ref[p] = (o > 0)
         ? k_native[j]
         : static_cast<Integer>(sizes[j] - 1 - k_native[j]);
   }

   return k_ref;
}

template <Integer Dim, size_t... Is >
GENDIL_HOST_DEVICE
std::array<Integer, Dim> NativeToReferenceIndex(
   const std::array<Integer, Dim>& k_native,
   const std::index_sequence<Is...>& sizes,
   const Permutation<Dim>& orientation)
{
   return NativeToReferenceIndex( k_native, std::array<size_t, Dim>{ Is... }, orientation );
}

}
