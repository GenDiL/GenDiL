// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <ostream>
#include <type_traits>
#include <utility>

#include "gendil/prelude.hpp"
#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

/**
 * @brief Represent a signed permutation of tensor axes.
 *
 * Positive entries preserve an axis direction and negative entries reverse it.
 * Absolute values use one-based reference-axis indices.
 *
 * @tparam Rank Number of tensor axes.
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

/** @brief Compile-time reference dimension of an orientation representation. */
template < class Orientation >
struct OrientationDimension;

template < size_t Rank >
struct OrientationDimension< Permutation< Rank > >
   : std::integral_constant< size_t, Rank >
{};

template < class Orientation >
inline constexpr size_t orientation_dimension_v =
   OrientationDimension< std::remove_cvref_t< Orientation > >::value;

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

/**
 * @brief Return whether a runtime orientation is a valid signed permutation.
 *
 * Every reference axis must occur exactly once, with either sign. Connectivity
 * construction is responsible for this precondition; hot geometry and DoF
 * paths use this helper only from debug assertions.
 */
template < size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool IsValidSignedPermutation(
   const Permutation< Rank > & orientation )
{
   std::array< bool, Rank > seen{};
   for ( size_t native_axis = 0; native_axis < Rank; ++native_axis )
   {
      const LocalIndex mapped_axis = orientation( native_axis );
      if ( mapped_axis == 0 )
      {
         return false;
      }

      const size_t reference_axis = static_cast< size_t >(
         mapped_axis > 0 ? mapped_axis - 1 : -mapped_axis - 1 );
      if ( reference_axis >= Rank || seen[ reference_axis ] )
      {
         return false;
      }
      seen[ reference_axis ] = true;
   }
   return true;
}

/** @brief Return whether a flat runtime orientation is structurally valid. */
template < size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool IsValidOrientation(
   const Permutation< Rank > & orientation )
{
   return IsValidSignedPermutation( orientation );
}

template < class Orientation >
struct IsRuntimePermutationOrientation : std::false_type
{};

template < size_t Rank >
struct IsRuntimePermutationOrientation< Permutation< Rank > > : std::true_type
{};

template < class Orientation >
inline constexpr bool is_runtime_permutation_orientation_v =
   IsRuntimePermutationOrientation<
      std::remove_cvref_t< Orientation > >::value;

/** @brief Insert a signed sub-permutation at a compile-time axis offset. */
template < size_t Offset, size_t Size, size_t Rank >
GENDIL_HOST_DEVICE
void Set( Permutation< Rank > & permutation,
          const Permutation< Size > & sub_permutation )
{
   static_assert(
      Offset + Size <= Rank,
      "Sub-permutation exceeds the destination permutation rank." );

   ConstexprLoop< Size >(
      [&](auto i) {
         const LocalIndex dim_index = sub_permutation( i );
         const LocalIndex offset = static_cast< LocalIndex >( Offset );
         permutation( Offset + i ) =
            dim_index > 0 ? offset + dim_index : dim_index - offset;
      }
   );
}

// Requires C++20
// template < size_t Offset, size_t Size, size_t Rank, Permutation< Size > sub_permutation >
// void Set( Permutation< Rank > & permutation, std::integral_constant< Permutation< Size >, sub_permutation > )
// {
//    Set< Offset >( permutation, sub_permutation, std::make_index_sequence<Size>{} );
// }

/** @brief Construct the unsigned identity permutation for the supplied axes. */
template < size_t Rank, size_t... Is >
constexpr auto MakeReferencePermutation( std::index_sequence< Is... > )
{
   return Permutation< Rank >( { (Is+1)... } );
}

/** @brief Construct an unsigned identity permutation of rank `Rank`. */
template < size_t Rank >
constexpr auto MakeReferencePermutation()
{
   return MakeReferencePermutation< Rank >( std::make_index_sequence< Rank >{} );
}

/**
 * @brief Return a flat runtime permutation unchanged.
 *
 * This overload lets diagnostics and interoperability code consume every
 * orientation representation through `FlattenOrientation`.
 */
template < size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr Permutation< Rank > FlattenOrientation(
   const Permutation< Rank > & orientation )
{
   return orientation;
}

/** @brief Write a flat permutation as its signed one-based axis entries. */
template < size_t Rank >
std::ostream & operator<<(
   std::ostream & os,
   const Permutation< Rank > & orientation )
{
   for ( size_t i = 0; i < Rank; ++i )
   {
      os << orientation( i ) << " ";
   }
   os << std::endl;
   return os;
}

/** @brief Extract a contiguous signed sub-permutation and remove its offset. */
template < size_t SubDim, size_t Dim >
GENDIL_HOST_DEVICE
auto GetSubPermutation(
   const Permutation< Dim > & orientation,
   size_t offset )
{
   Permutation< SubDim > sub_permutation{};
   for ( size_t i = 0; i < SubDim; ++i )
   {
      sub_permutation( i ) = orientation( offset + i ) > 0
         ? orientation( offset + i ) - static_cast< LocalIndex >( offset )
         : orientation( offset + i ) + static_cast< LocalIndex >( offset );
   }

   return sub_permutation;
}

} // namespace gendil
