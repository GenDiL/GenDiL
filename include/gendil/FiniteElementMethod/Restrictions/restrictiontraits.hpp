// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

#include <limits>
#include <type_traits>

namespace gendil {

inline constexpr size_t dynamic_restriction_entry_count =
   std::numeric_limits< size_t >::max();

template < typename Restriction >
inline constexpr size_t static_restriction_entry_count_v =
   dynamic_restriction_entry_count;

/**
 * @brief Select the restriction associated with one compile-time component.
 *
 * Scalar restrictions are treated as one-component restrictions and therefore
 * accept only component zero. Composite restriction types specialize this
 * trait to expose their child restriction types.
 */
template < typename Restriction, size_t Component >
struct RestrictionComponent
{
   static_assert(
      Component == 0,
      "Scalar restrictions only have component zero." );
   using type = std::remove_cvref_t< Restriction >;
};

/** @brief Restriction type associated with compile-time component @c Component. */
template < typename Restriction, size_t Component >
using restriction_component_t = typename RestrictionComponent<
   std::remove_cvref_t< Restriction >,
   Component >::type;

/**
 * @brief Compile-time identity of one native component-local restriction row.
 *
 * This tag lets coordinate assembly characterize rows whose exact stencil
 * cardinality varies within one restriction type. The local DoF index is
 * flattened within one component after orientation has selected native
 * ordering; it is not an algebraic coordinate or a restriction-entry index.
 * Restrictions with uniform cardinality need no specialization: the default
 * below forwards to the restriction-wide count and therefore retains
 * closed-form placement.
 */
template < size_t Component, GlobalIndex ComponentLocalDofIndex >
struct NativeRestrictionRowTag
{
   static constexpr size_t component = Component;
   static constexpr GlobalIndex local_dof_index = ComponentLocalDofIndex;
};

template < typename Restriction, typename NativeRowTag >
inline constexpr size_t static_restriction_entry_count_for_v =
   static_restriction_entry_count_v< std::remove_cvref_t< Restriction > >;

template < typename Restriction >
inline constexpr bool restriction_may_share_global_dofs_v = true;

/**
 * @brief Whether each individual row visits every algebraic coordinate at
 * most once, including explicit zero-weight entries.
 *
 * This is independent of overlap between different rows. Exact zero- and
 * one-entry rows satisfy the property by construction. Multi-entry and
 * dynamic custom restrictions remain conservative until they explicitly
 * provide the fact.
 */
template < typename Restriction >
inline constexpr bool restriction_rows_have_unique_coordinates_v =
   static_restriction_entry_count_v<
      std::remove_cvref_t< Restriction > > !=
      dynamic_restriction_entry_count &&
   static_restriction_entry_count_v<
      std::remove_cvref_t< Restriction > > <= 1;

/** @brief Whether the current row-parallel scatter can avoid atomics. */
template < typename Restriction >
inline constexpr bool restriction_scatter_is_conflict_free_v =
   restriction_rows_have_unique_coordinates_v<
      std::remove_cvref_t< Restriction > > &&
   !restriction_may_share_global_dofs_v<
      std::remove_cvref_t< Restriction > >;

/**
 * @brief Whether every restriction row aliases one unit-weight algebraic
 * storage entry.
 *
 * This capability permits a reference-returning element view. It does not
 * imply injectivity or make an unguarded scatter operation safe.
 */
template < typename Restriction >
struct restriction_supports_element_reference_view : std::false_type { };

template < typename Restriction >
inline constexpr bool restriction_supports_element_reference_view_v =
   restriction_supports_element_reference_view<
      std::remove_cvref_t< Restriction > >::value;

} // namespace gendil
