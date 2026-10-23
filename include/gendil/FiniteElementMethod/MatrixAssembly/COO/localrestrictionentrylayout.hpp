// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Element-local restriction-entry layout used by RawCOO assembly.
 *
 * A finite-element local DoF identifies one row of its algebraic restriction.
 * That row can visit zero, one, or several algebraic entries. RawCOO stores one
 * coordinate contribution for every pair of visited test and trial entries,
 * so each side of a local operator block needs a flat ordinal for those visits.
 *
 * This header defines that element-local ordering. Components are concatenated
 * in component-major order. Within each component, native local DoF rows retain
 * their flattened tensor order, and entries retain the deterministic order in
 * which `ForEachRestrictionEntry` visits them. If native row @f$r@f$ contains
 * @f$n_r@f$ entries, its entry @f$k@f$ has component-local ordinal
 *
 * @f[
 *    p_r + k, \qquad p_r = \sum_{t<r} n_t.
 * @f]
 *
 * Uniform restrictions use the closed form @f$rn+k@f$ and instantiate no
 * per-row count or prefix table. Only genuinely nonuniform native rows use
 * compile-time count and prefix tables. Runtime facet orientation must select
 * the native local row before this ordering is queried.
 *
 * These ordinals describe one axis of an element-local RawCOO block. They are
 * neither algebraic/global DoF coordinates nor final RawCOO triplet indices.
 */

#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

#include <array>
#include <limits>
#include <type_traits>
#include <utility>

namespace gendil {

namespace details {

/// @brief Number of scalar components represented by a finite-element space.
template < typename Space >
inline constexpr size_t space_component_count_v = []
{
   using ShapeFunctions = finite_element_space_shape_functions_t< Space >;
   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      return static_cast< size_t >( ShapeFunctions::vector_dim );
   }
   else
   {
      return size_t{ 1 };
   }
}();

/// @brief Number of native local DoF rows in one component.
template < typename Space, size_t Component >
inline constexpr GlobalIndex component_native_row_count_v = []
{
   using ShapeFunctions = finite_element_space_shape_functions_t< Space >;
   using Shape = component_dof_shape_t< ShapeFunctions, Component >;
   return static_cast< GlobalIndex >( Product( Shape{} ) );
}();

/// @brief Build the exact restriction-entry count for every native row.
template < typename Space, size_t Component, GlobalIndex... Row >
consteval auto MakeNativeRowCountTable( std::integer_sequence< GlobalIndex, Row... > )
{
   using SpaceType = std::remove_cvref_t< Space >;
   using Restriction = typename SpaceType::restriction_type;
   return std::array< size_t, sizeof...( Row ) >{
      static_restriction_entry_count_for_v<
         Restriction,
         NativeRestrictionRowTag< Component, Row > >... };
}

/// @brief Return the compile-time native-row count table for one component.
template < typename Space, size_t Component >
consteval auto NativeRowCountTable()
{
   return MakeNativeRowCountTable< Space, Component >(
      std::make_integer_sequence<
         GlobalIndex,
         component_native_row_count_v< Space, Component > >{} );
}

/// @brief Report whether every native row in a component has a static count.
template < typename Space, size_t Component >
consteval bool ComponentHasStaticNativeRowCounts()
{
   constexpr auto counts = NativeRowCountTable< Space, Component >();
   for ( const size_t count : counts )
   {
      if ( count == dynamic_restriction_entry_count )
      {
         return false;
      }
   }
   return true;
}

/// @brief Sum all restriction entries in one component, checking overflow.
///
/// Returns the dynamic-count sentinel if any native row lacks an exact static
/// entry count.
template < typename Space, size_t Component >
consteval GlobalIndex ExpandedComponentRowCount()
{
   constexpr auto result = []
   {
      struct Result { GlobalIndex value; bool valid; };
      constexpr auto counts = NativeRowCountTable< Space, Component >();
      GlobalIndex total = 0;
      for ( const size_t count : counts )
      {
         if ( count == dynamic_restriction_entry_count )
         {
            return Result{ static_cast< GlobalIndex >(
               dynamic_restriction_entry_count ), true };
         }
         if (
            count > static_cast< size_t >(
               std::numeric_limits< GlobalIndex >::max() ) ||
            total > std::numeric_limits< GlobalIndex >::max() -
               static_cast< GlobalIndex >( count ) )
         {
            return Result{ 0, false };
         }
         total += static_cast< GlobalIndex >( count );
      }
      return Result{ total, true };
   }();
   static_assert( result.valid,
      "Expanded RawCOO restriction-row count overflows GlobalIndex." );
   return result.value;
}

/// @brief Report whether all components have exact native-row entry counts.
template < typename Space, size_t... Component >
consteval bool HasStaticNativeRowCounts( std::index_sequence< Component... > )
{
   return ( ComponentHasStaticNativeRowCounts< Space, Component >() && ... );
}

/// @brief Sum the restriction-entry counts of all space components.
template < typename Space, size_t... Component >
consteval GlobalIndex ExpandedLocalRowCount( std::index_sequence< Component... > )
{
   return ( GlobalIndex{ 0 } + ... +
      ExpandedComponentRowCount< Space, Component >() );
}

/// @brief Sum entry counts of the components preceding @p Component.
template < typename Space, size_t Component, size_t... Previous >
consteval GlobalIndex ExpandedComponentOffset_impl(
   std::index_sequence< Previous... > )
{
   return ( GlobalIndex{ 0 } + ... +
      ExpandedComponentRowCount< Space, Previous >() );
}

/// @brief Return one component's offset in component-major entry ordering.
template < typename Space, size_t Component >
consteval GlobalIndex ExpandedComponentOffset()
{
   return ExpandedComponentOffset_impl< Space, Component >(
      std::make_index_sequence< Component >{} );
}

/// @brief Build exclusive native-row entry prefixes for a nonuniform component.
template < typename Space, size_t Component >
consteval auto NativeRowPrefixTable()
{
   constexpr auto result = []
   {
      constexpr auto counts = NativeRowCountTable< Space, Component >();
      struct Result
      {
         std::array< GlobalIndex,
            component_native_row_count_v< Space, Component > > values{};
         bool valid = true;
      } result;
      GlobalIndex prefix = 0;
      for ( size_t row = 0; row < counts.size(); ++row )
      {
         result.values[row] = prefix;
         if (
            counts[row] > static_cast< size_t >(
               std::numeric_limits< GlobalIndex >::max() ) ||
            prefix > std::numeric_limits< GlobalIndex >::max() -
               static_cast< GlobalIndex >( counts[row] ) )
         {
            result.valid = false;
            return result;
         }
         prefix += static_cast< GlobalIndex >( counts[row] );
      }
      return result;
   }();
   static_assert( result.valid,
      "Expanded RawCOO restriction-row prefix overflows GlobalIndex." );
   return result.values;
}

} // namespace details

/**
 * @brief Whether every native local restriction row has an exact static count.
 *
 * RawCOO requires this property to plan its coordinate storage before kernel
 * execution. Matrix-free and SGBSR application do not require it.
 */
template < typename Space >
inline constexpr bool has_static_expanded_restriction_rows_v =
   details::HasStaticNativeRowCounts< std::remove_cvref_t< Space > >(
      std::make_index_sequence<
         details::space_component_count_v< std::remove_cvref_t< Space > > >{} );

/**
 * @brief Total restriction-entry count along one element-local RawCOO axis.
 *
 * The count includes every component, native local DoF row, zero-weight entry,
 * and duplicate coordinate visit. It is not a global DoF count or the number
 * of test/trial pairs in the complete local RawCOO block.
 */
template < typename Space >
consteval GlobalIndex CountElementRestrictionEntries()
{
   using SpaceType = std::remove_cvref_t< Space >;
   static_assert(
      has_static_expanded_restriction_rows_v< SpaceType >,
      "RawCOO requires an exact compile-time restriction-entry count for "
      "every native local row." );
   return details::ExpandedLocalRowCount< SpaceType >(
      std::make_index_sequence<
         details::space_component_count_v< SpaceType > >{} );
}

/**
 * @brief Return the exact number of entries in one native local DoF row.
 *
 * Uniform restrictions return their restriction-wide compile-time count.
 * Nonuniform restrictions select the exact count from compile-time metadata
 * using the flattened native local DoF row.
 */
template <
   typename Space,
   size_t Component,
   Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex CountLocalDofRestrictionEntries(
   const Space &,
   std::integral_constant< size_t, Component > component,
   const std::array< GlobalIndex, Dim > & indices )
{
   using SpaceType = std::remove_cvref_t< Space >;
   using ShapeFunctions =
      finite_element_space_shape_functions_t< SpaceType >;
   using Restriction = restriction_component_t<
      typename SpaceType::restriction_type,
      Component >;
   constexpr size_t uniform_count =
      static_restriction_entry_count_v< Restriction >;

   if constexpr ( uniform_count != dynamic_restriction_entry_count )
   {
      return static_cast< GlobalIndex >( uniform_count );
   }
   else
   {
      static_assert(
         details::ComponentHasStaticNativeRowCounts<
            SpaceType,
            Component >(),
         "This restriction row has dynamic cardinality and cannot be emitted "
         "into statically planned RawCOO storage." );
      constexpr auto counts =
         details::NativeRowCountTable< SpaceType, Component >();
      const GlobalIndex local_row =
         FlattenComponentLocalDof< ShapeFunctions >( component, indices );
      return static_cast< GlobalIndex >( counts[local_row] );
   }
}

/**
 * @brief Flatten one visited restriction entry into a local RawCOO-axis ordinal.
 *
 * The supplied indices identify a native local DoF row and @p entry_ordinal
 * identifies one visit within that row. Component offsets are included in the
 * result. Uniform restrictions use `component_offset + row * count + entry`;
 * nonuniform restrictions use the compile-time exclusive prefix of the row.
 */
template <
   typename Space,
   size_t Component,
   Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex FlattenLocalRestrictionEntry(
   const Space &,
   std::integral_constant< size_t, Component > component,
   const std::array< GlobalIndex, Dim > & indices,
   const GlobalIndex entry_ordinal )
{
   using SpaceType = std::remove_cvref_t< Space >;
   using ShapeFunctions =
      finite_element_space_shape_functions_t< SpaceType >;
   using Restriction = restriction_component_t<
      typename SpaceType::restriction_type,
      Component >;
   constexpr GlobalIndex component_offset =
      details::ExpandedComponentOffset< SpaceType, Component >();
   const GlobalIndex local_row =
      FlattenComponentLocalDof< ShapeFunctions >( component, indices );
   constexpr size_t uniform_count =
      static_restriction_entry_count_v< Restriction >;

   if constexpr ( uniform_count != dynamic_restriction_entry_count )
   {
      return component_offset +
         local_row * static_cast< GlobalIndex >( uniform_count ) +
         entry_ordinal;
   }
   else
   {
      static_assert(
         details::ComponentHasStaticNativeRowCounts<
            SpaceType,
            Component >(),
         "This restriction row has dynamic cardinality and cannot be emitted "
         "into statically planned RawCOO storage." );
      constexpr auto prefixes =
         details::NativeRowPrefixTable< SpaceType, Component >();
      return component_offset + prefixes[local_row] + entry_ordinal;
   }
}

} // namespace gendil
