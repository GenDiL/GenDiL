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

template < typename Restriction >
inline constexpr bool restriction_may_share_global_dofs_v = true;

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
