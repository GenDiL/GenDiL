// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Describe a tensor's dimensions and query its number of logical entries,
 * independently of how its values are stored.
 *
 * An extent is an axis length; rank is the number of axes. Axes are numbered
 * from zero. For example, a 2-by-3 shape has rank two and six logical entries:
 *
 * @code
 * DynamicExtents<2> shape{2, 3};
 * auto first_dimension = GetExtentValue<0>(shape); // 2
 * auto entry_count = GetLogicalSize(shape);       // 6
 * @endcode
 *
 * Choose StaticExtents for dimensions known at compile time, DynamicExtents
 * for lengths supplied at runtime, or NoExtents when lengths are intentionally
 * unavailable. An unknown length is not a zero length.
 *
 * The same queries work on layouts that expose their dimensions. Shape alone
 * says nothing about storage strides, contiguity, allocation size, or a
 * finite-element field's algebraic DoF extent.
 */

#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/types.hpp"

#include <array>
#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

namespace gendil {

namespace detail {

/** Extents are counts, not Boolean flags or rounded real numbers. */
template < class T >
concept ExtentIndex = std::integral< std::remove_cvref_t< T > > &&
   !std::same_as< std::remove_cvref_t< T >, bool >;

/** Prove a type-known shape product fits before doing unsigned arithmetic. */
template < size_t Rank >
consteval bool StaticExtentProductFits( const std::array< GlobalIndex, Rank > & values )
{
   for ( const auto value : values )
      if ( value == 0 ) { return true; }
   GlobalIndex product = 1;
   for ( const auto value : values )
   {
      if ( value > std::numeric_limits< GlobalIndex >::max() / product )
         return false;
      product *= value;
   }
   return true;
}

/** Convert valid metadata; CPU debug checks precede narrowing/sign conversion. */
template < ExtentIndex To, ExtentIndex From >
GENDIL_HOST_DEVICE
constexpr To CheckedExtentCast( const From value )
{
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
   if constexpr ( std::signed_integral< From > )
   {
      GENDIL_ASSERT( value >= 0, "View extents must be nonnegative." );
   }
   using UnsignedFrom = std::make_unsigned_t< From >;
   const auto positive = static_cast< UnsignedFrom >( value );
   if constexpr ( std::numeric_limits< UnsignedFrom >::digits >
                  std::numeric_limits< To >::digits )
   {
      GENDIL_ASSERT(
         positive <= static_cast< UnsignedFrom >(
            std::numeric_limits< To >::max() ),
         "View extent is not representable in the destination index type." );
   }
#endif
   return static_cast< To >( value );
}

/**
 * CPU debug checks every axis, including after a zero length. Release/device
 * execution computes only the product. An empty axis list returns one;
 * a zero factor produces zero. Nonempty products must fit in GlobalIndex.
 */
template < ExtentIndex Index, size_t Rank >
GENDIL_HOST_DEVICE
constexpr GlobalIndex CheckedExtentProduct(
   const std::array< Index, Rank > & values )
{
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
   bool empty = false;
   for ( const auto value : values )
   {
      // Every axis must be valid, including axes following an empty one.
      empty = ( CheckedExtentCast< GlobalIndex >( value ) == 0 ) || empty;
   }
   if ( empty )
   {
      return 0;
   }
#endif
   GlobalIndex product = 1;
   for ( const auto value : values )
   {
      GENDIL_ASSERT(
         product == 0 || static_cast< GlobalIndex >( value ) <=
            std::numeric_limits< GlobalIndex >::max() / product,
         "View extent product overflows GlobalIndex." );
      product *= static_cast< GlobalIndex >( value );
   }
   return product;
}

} // namespace detail

/** @brief Select a zero-based logical axis without storing a runtime index. */
template < size_t Axis >
using AxisTag = std::integral_constant< size_t, Axis >;

/**
 * @brief Describe a tensor's rank without retaining its axis lengths.
 *
 * Use this when lengths need not be available to callers. No axis queries
 * are provided: a positive-rank NoExtents has an unknown shape, not an empty
 * one. NoExtents<0> describes a scalar and has logical size one.
 */
template < size_t Rank >
struct NoExtents
{
   static constexpr size_t rank = Rank;
};

/**
 * @brief Describe a tensor whose dimensions are known at compile time.
 *
 * Lengths are encoded in the type, with no runtime dimension storage. Axis
 * queries return integral constants; @c logical_size is their product,
 * checked for representability at compile time. Zero lengths are permitted.
 * StaticExtents<> describes one scalar, not an empty tensor.
 */
template < GlobalIndex... Values >
struct StaticExtents
{
   static constexpr size_t rank = sizeof...( Values );
   static_assert( detail::StaticExtentProductFits(
      std::array< GlobalIndex, rank >{ Values... } ),
      "Static extent product overflows GlobalIndex." );
   static constexpr GlobalIndex logical_size = ( GlobalIndex{ 1 } * ... * Values );

   /** @brief Return the selected type-known length as an integral constant. */
   template < size_t Axis >
      requires ( Axis < rank )
   GENDIL_HOST_DEVICE
   static constexpr auto Get( AxisTag< Axis > )
   {
      constexpr std::array< GlobalIndex, rank > values{ Values... };
      return std::integral_constant< GlobalIndex, values[Axis] >{};
   }

   /** @brief Convenience spelling of @c Get(AxisTag<Axis>{}). */
   template < size_t Axis >
      requires ( Axis < rank )
   GENDIL_HOST_DEVICE
   static constexpr auto Get()
   {
      return Get( AxisTag< Axis >{} );
   }
};

/** @brief Convert a std::index_sequence to a checked StaticExtents type. */
template < class Sequence >
struct static_extents_from_sequence;

template < size_t... Values >
struct static_extents_from_sequence< std::index_sequence< Values... > >
{
   static_assert( ( std::in_range< GlobalIndex >( Values ) && ... ),
      "Static extent is not representable in GlobalIndex." );
   using type = StaticExtents< static_cast< GlobalIndex >( Values )... >;
};

/** @brief Static shape corresponding to the lengths in an index sequence. */
template < class Sequence >
using static_extents_from_sequence_t =
   typename static_extents_from_sequence< Sequence >::type;

/**
 * @brief Describe a tensor using dimension lengths supplied at runtime.
 *
 * Supply one length per axis or an integral array. The shape stores exactly
 * Rank GlobalIndex lengths in @c values. Zero lengths are permitted, though
 * a layout using the shape may require positive lengths.
 *
 * Boolean and floating-point inputs are rejected. Integral values must be
 * nonnegative, and each length and their product must fit in GlobalIndex.
 * CPU debug construction checks these preconditions; release and device
 * execution trust them. Callers modifying @c values must preserve validity.
 */
template < size_t Rank >
struct DynamicExtents
{
   static constexpr size_t rank = Rank;

   std::array< GlobalIndex, Rank > values{};

   /** @brief Initialize all lengths to zero; rank zero still has size one. */
   GENDIL_HOST_DEVICE
   constexpr DynamicExtents() = default;

   /** @brief Copy an integral array of exactly Rank valid lengths. */
   template < detail::ExtentIndex InputIndex >
   GENDIL_HOST_DEVICE
   constexpr explicit DynamicExtents(
      const std::array< InputIndex, Rank > & input )
   {
      for ( size_t axis = 0; axis < Rank; ++axis )
      {
         values[axis] = detail::CheckedExtentCast< GlobalIndex >( input[axis] );
      }
      Validate();
   }

   /** @brief Supply one valid integral argument per axis; use {} for rank zero. */
   template < class... Values >
      requires (
         Rank > 0 && sizeof...( Values ) == Rank &&
         ( detail::ExtentIndex< Values > && ... ) )
   GENDIL_HOST_DEVICE
   constexpr explicit DynamicExtents( Values... input )
      : values{ detail::CheckedExtentCast< GlobalIndex >( input )... }
   {
      Validate();
   }

   /** @brief Read one stored length as a GlobalIndex value. */
   template < size_t Axis >
      requires ( Axis < rank )
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Get( AxisTag< Axis > ) const
   {
      return values[Axis];
   }

   /** @brief Convenience spelling of @c Get(AxisTag<Axis>{}). */
   template < size_t Axis >
      requires ( Axis < rank )
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Get() const
   {
      return Get( AxisTag< Axis >{} );
   }

private:
   /** Validate the complete shape without doing unused release/device work. */
   GENDIL_HOST_DEVICE GENDIL_INLINE
   constexpr void Validate() const
   {
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
      (void)detail::CheckedExtentProduct( values );
#endif
   }
};

namespace detail {

/** Recognize type-encoded values without requiring a member on other types. */
template < class T >
struct is_integral_constant : std::false_type { };

template < class T, T Value >
struct is_integral_constant< std::integral_constant< T, Value > >
   : std::true_type { };

/** Ignore cv/ref qualifiers when recognizing an integral constant. */
template < class T >
inline constexpr bool is_integral_constant_v =
   is_integral_constant< std::remove_cvref_t< T > >::value;

/** Extent queries return an integral count or its compile-time equivalent. */
template < class Query >
inline constexpr bool is_extent_query_v = []
{
   using Type = std::remove_cvref_t< Query >;
   if constexpr ( is_integral_constant_v< Type > )
      return ExtentIndex< typename Type::value_type >;
   else
      return ExtentIndex< Type >;
}();

/** Check one axis before folding, avoiding a requires-expression over a pack. */
template < class Object, size_t Axis >
concept HasExtentQuery = requires( const Object & object )
{
   object.Get( AxisTag< Axis >{} );
   requires is_extent_query_v< decltype( object.Get( AxisTag< Axis >{} ) ) >;
};

/** Require a supported query result for each axis in the object's rank. */
template < class Object, size_t... Axis >
consteval bool HasCompleteExtentsImpl( std::index_sequence< Axis... > )
{
   return ( HasExtentQuery< Object, Axis > && ... );
}

} // namespace detail

/**
 * @brief Query an axis length while preserving compile-time information.
 *
 * Static axes return integral constants, useful when a dimension determines
 * a template argument or a compile-time calculation:
 *
 * @code
 * constexpr StaticExtents<2, 3> fixed;
 * static_assert(decltype(GetExtent<0>(fixed))::value == 2);
 * @endcode
 *
 * This adapter calls @c object.Get(AxisTag<Axis>{}) on the const object and
 * preserves its result type: a non-Boolean integral value, directly or as
 * std::integral_constant.
 * An individual query needs no @c rank. Unlike GetExtentValue, this accessor
 * does not convert or validate the returned numeric value.
 */
template < size_t Axis, class Object >
   requires requires( const Object & object )
   {
      object.Get( AxisTag< Axis >{} );
      requires detail::is_extent_query_v<
         decltype( object.Get( AxisTag< Axis >{} ) ) >;
   }
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetExtent( const Object & object )
{
   return object.Get( AxisTag< Axis >{} );
}

/**
 * @brief Obtain an axis length as an ordinary GlobalIndex integer.
 *
 * Both integral constants and runtime query values are accepted. Type-known
 * invalid values fail compilation; runtime values are CPU-debug-checked.
 */
template < size_t Axis, class Object >
   requires requires( const Object & object )
   {
      GetExtent< Axis >( object );
   }
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetExtentValue( const Object & object )
{
   const auto extent = GetExtent< Axis >( object );
   if constexpr ( detail::is_integral_constant_v< decltype( extent ) > )
   {
      static_assert( std::in_range< GlobalIndex >( +decltype( extent )::value ),
         "Static extent is not representable in GlobalIndex." );
      return static_cast< GlobalIndex >( decltype( extent )::value );
   }
   else
   {
      return detail::CheckedExtentCast< GlobalIndex >( extent );
   }
}

/** @brief Whether the length of one selected axis can be queried. */
template < size_t Axis, class Object >
inline constexpr bool has_extent_v = requires( const Object & object )
{
   GetExtent< Axis >( object );
};

/** @brief Require a queryable axis length, without validating its value. */
template < size_t Axis, class Object >
concept HasExtent = has_extent_v< Axis, Object >;

/** @brief Whether a compile-time rank and lengths for every axis can be queried. */
template < class Object >
inline constexpr bool has_complete_extents_v =
   []
   {
      using Type = std::remove_cvref_t< Object >;
      if constexpr ( requires
      {
         typename std::integral_constant< size_t, Type::rank >;
      } )
      {
         return detail::HasCompleteExtentsImpl< Type >(
            std::make_index_sequence< Type::rank >{} );
      }
      else
      {
         return false;
      }
   }();

/** @brief Require a queryable complete shape, independent of storage layout. */
template < class Object >
concept HasCompleteExtents = has_complete_extents_v< Object >;

/**
 * @brief Obtain the number of logical tensor entries.
 *
 * Multiplies all axis lengths and returns a GlobalIndex, usable in constant
 * expressions when the input permits. A zero-length axis gives zero entries;
 * rank zero describes a scalar and returns one.
 *
 * Requires a complete, valid shape. CPU debug checks every axis and the
 * product; release and device execution assume valid, representable values.
 * This count says nothing about the storage span or algebraic vector size.
 */
template < class Object >
   requires has_complete_extents_v< Object >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetLogicalSize( const Object & object )
{
   using Type = std::remove_cvref_t< Object >;
   return [&]< size_t... Axis >( std::index_sequence< Axis... > )
   {
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
      return detail::CheckedExtentProduct(
         std::array< GlobalIndex, Type::rank >{
            GetExtentValue< Axis >( object )... } );
#else
      return ( GlobalIndex{ 1 } * ... *
         GetExtentValue< Axis >( object ) );
#endif
   }( std::make_index_sequence< Type::rank >{} );
}

} // namespace gendil
