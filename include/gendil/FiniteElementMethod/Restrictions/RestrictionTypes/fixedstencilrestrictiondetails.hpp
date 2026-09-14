// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Implementation details for fixed-stencil restrictions.
 *
 * This header recognizes and validates the compile-time stencil definition,
 * computes its row metadata, and implements statically unrolled row
 * visitation. Users should include `fixedstencilrestriction.hpp` instead.
 */

#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/Utilities/arraytraits.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil::details {

/** Recognizes the static arrays required by a fixed-stencil definition. */
template < class Stencil >
concept FixedStencilDefinition =
   requires
   {
      Stencil::row_entry_counts;
      Stencil::entry_weights;
   } &&
   is_std_array_v< decltype( Stencil::row_entry_counts ) > &&
   is_std_array_v< decltype( Stencil::entry_weights ) >;

template < class Stencil >
using fixed_stencil_row_counts_t =
   std::remove_cvref_t< decltype( Stencil::row_entry_counts ) >;

template < class Stencil >
using fixed_stencil_weights_t =
   std::remove_cvref_t< decltype( Stencil::entry_weights ) >;

template < class Stencil >
using fixed_stencil_weight_t =
   typename fixed_stencil_weights_t< Stencil >::value_type;

template < class Weight >
consteval bool FixedStencilArithmeticWeightIsFinite( const Weight weight )
{
   if constexpr ( std::floating_point< Weight > )
   {
      return weight == weight &&
         weight != std::numeric_limits< Weight >::infinity() &&
         weight != -std::numeric_limits< Weight >::infinity();
   }
   else
   {
      return true;
   }
}

template < class Stencil >
consteval bool FixedStencilWeightsAreFinite()
{
   using Weight = fixed_stencil_weight_t< Stencil >;
   if constexpr ( ArithmeticRestrictionWeight< Weight > )
   {
      for ( const Weight weight : Stencil::entry_weights )
      {
         if ( !FixedStencilArithmeticWeightIsFinite( weight ) )
         {
            return false;
         }
      }
   }
   return true;
}

/** Count all restriction entries represented for one element. */
template < class Stencil >
consteval std::size_t FixedStencilEntriesPerElement()
{
   std::size_t total = 0;
   for ( const std::size_t count : Stencil::row_entry_counts )
   {
      if ( count > std::numeric_limits< std::size_t >::max() - total )
      {
         throw "Fixed-stencil row-entry total overflows size_t.";
      }
      total += count;
   }
   return total;
}

/** Return the exact entry count of one native local DoF row. */
template < class Stencil, std::size_t Row >
consteval std::size_t FixedStencilRowEntryCount()
{
   static_assert(
      Row < std::tuple_size_v< fixed_stencil_row_counts_t< Stencil > >,
      "Fixed-stencil native row is out of range." );
   return Stencil::row_entry_counts[Row];
}

/** Return one native row's prefix in the element index and weight arrays. */
template < class Stencil, std::size_t Row >
consteval std::size_t FixedStencilRowEntryOffset()
{
   static_assert(
      Row <= std::tuple_size_v< fixed_stencil_row_counts_t< Stencil > >,
      "Fixed-stencil native row is out of range." );
   std::size_t offset = 0;
   for ( std::size_t row = 0; row < Row; ++row )
   {
      const std::size_t count = Stencil::row_entry_counts[row];
      if ( count > std::numeric_limits< std::size_t >::max() - offset )
      {
         throw "Fixed-stencil row-entry prefix overflows size_t.";
      }
      offset += count;
   }
   return offset;
}

/**
 * Return a common row count, or the dynamic sentinel if rows differ.
 *
 * The sentinel only means that no restriction-wide uniform count exists.
 * Every native row still has an exact compile-time count.
 */
template < class Stencil >
consteval std::size_t FixedStencilUniformEntryCount()
{
   constexpr std::size_t num_rows =
      std::tuple_size_v< fixed_stencil_row_counts_t< Stencil > >;
   if constexpr ( num_rows == 0 )
   {
      return 0;
   }
   else
   {
      constexpr std::size_t count = Stencil::row_entry_counts[0];
      for ( std::size_t row = 1; row < num_rows; ++row )
      {
         if ( Stencil::row_entry_counts[row] != count )
         {
            return dynamic_restriction_entry_count;
         }
      }
      return count;
   }
}

/** Whether cardinality alone proves uniqueness within every row. */
template < class Stencil >
consteval bool FixedStencilRowsHaveAtMostOneEntry()
{
   for ( const std::size_t count : Stencil::row_entry_counts )
   {
      if ( count > 1 )
      {
         return false;
      }
   }
   return true;
}

template < class Stencil >
consteval bool FixedStencilWeightsSupportRestrictionOperations()
{
   using Weight = fixed_stencil_weight_t< Stencil >;
   if constexpr ( std::same_as< std::remove_cv_t< Weight >, bool > )
   {
      return false;
   }
   else
   {
      return requires( const Weight & weight, const Real value )
      {
         { ApplyRestrictionWeight( weight, value ) }
            -> std::convertible_to< Real >;
         { ApplyAdjointRestrictionWeight( weight, value ) }
            -> std::convertible_to< Real >;
      };
   }
}

/** Validate a zero-state stencil definition against one scalar DoF shape. */
template < class Stencil, StaticDofShape DofShape >
consteval bool ValidateFixedStencilForDofShape()
{
   static_assert(
      std::is_empty_v< Stencil >,
      "Fixed-stencil definitions must be zero-state types containing only static compile-time metadata." );
   constexpr std::size_t num_rows =
      std::tuple_size_v< fixed_stencil_row_counts_t< Stencil > >;
   constexpr std::size_t num_weights =
      std::tuple_size_v< fixed_stencil_weights_t< Stencil > >;
   static_assert(
      std::same_as<
         typename fixed_stencil_row_counts_t< Stencil >::value_type,
         std::size_t >,
      "Fixed-stencil row-entry counts must use std::size_t." );
   static_assert(
      num_rows == static_cast< std::size_t >( Product( DofShape{} ) ),
      "Fixed-stencil definition must provide one row-entry count per native finite-element DoF." );
   static_assert(
      FixedStencilEntriesPerElement< Stencil >() == num_weights,
      "Fixed-stencil definition row-entry counts must sum to the number of entry weights." );
   static_assert(
      FixedStencilWeightsSupportRestrictionOperations< Stencil >(),
      "Fixed-stencil weights must support restriction gather and adjoint application; raw bool weights are not supported." );
   static_assert(
      FixedStencilWeightsAreFinite< Stencil >(),
      "Fixed-stencil arithmetic weights must be finite." );
   static_assert(
      FixedStencilEntriesPerElement< Stencil >() <=
         std::numeric_limits< GlobalIndex >::max(),
      "Fixed-stencil entries per element exceed GlobalIndex." );
   return true;
}

template < class Index >
GENDIL_HOST_DEVICE
constexpr GlobalIndex FixedStencilAlgebraicIndex( const Index index )
{
   static_assert(
      std::integral< Index > &&
         !std::same_as< std::remove_cv_t< Index >, bool >,
      "Fixed-stencil algebraic indices must use a non-Boolean integral type." );
   if constexpr ( std::signed_integral< Index > )
   {
      GENDIL_ASSERT(
         index >= 0,
         "FixedStencilRestriction contains a negative algebraic index." );
   }
   if constexpr ( sizeof( Index ) > sizeof( GlobalIndex ) )
   {
      using UnsignedIndex = std::make_unsigned_t< Index >;
      GENDIL_ASSERT(
         static_cast< UnsignedIndex >( index ) <=
            std::numeric_limits< GlobalIndex >::max(),
         "FixedStencilRestriction algebraic index exceeds GlobalIndex." );
   }
   return static_cast< GlobalIndex >( index );
}

template < std::size_t Row, class Restriction, class Visitor, std::size_t... Entry >
GENDIL_HOST_DEVICE
constexpr void VisitFixedStencilRowEntries(
   const Restriction & restriction,
   const GlobalIndex element,
   Visitor & visitor,
   std::index_sequence< Entry... > )
{
   using Stencil = typename Restriction::stencil_type;
   constexpr std::size_t row_offset =
      FixedStencilRowEntryOffset< Stencil, Row >();
   constexpr GlobalIndex entries_per_element = static_cast< GlobalIndex >(
      FixedStencilEntriesPerElement< Stencil >() );
   const GlobalIndex element_offset = element * entries_per_element;

   ( [&]
      {
         constexpr auto weight = Stencil::entry_weights[row_offset + Entry];
         const GlobalIndex map_entry =
            element_offset + static_cast< GlobalIndex >( row_offset + Entry );
         GENDIL_ASSERT(
            map_entry < restriction.index_count,
            "FixedStencilRestriction index-map entry is out of bounds." );
         const GlobalIndex algebraic_dof = FixedStencilAlgebraicIndex(
            restriction.indices[map_entry] );
         GENDIL_ASSERT(
            algebraic_dof < restriction.algebraic_dof_extent,
            "FixedStencilRestriction algebraic index is out of bounds." );
         visitor( algebraic_dof, weight );
      }(), ... );
}

template < std::size_t Row, class Restriction, class Visitor >
GENDIL_HOST_DEVICE
constexpr void DispatchFixedStencilRow(
   const Restriction & restriction,
   const GlobalIndex element,
   const GlobalIndex native_row,
   Visitor & visitor )
{
   using Stencil = typename Restriction::stencil_type;
   constexpr std::size_t num_rows =
      std::tuple_size_v< fixed_stencil_row_counts_t< Stencil > >;
   if constexpr ( Row < num_rows )
   {
      if ( native_row == Row )
      {
         VisitFixedStencilRowEntries< Row >(
            restriction,
            element,
            visitor,
            std::make_index_sequence<
               FixedStencilRowEntryCount< Stencil, Row >() >{} );
      }
      else
      {
         DispatchFixedStencilRow< Row + 1 >(
            restriction,
            element,
            native_row,
            visitor );
      }
   }
}

} // namespace gendil::details
