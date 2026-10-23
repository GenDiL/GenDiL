// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Type traits for recognizing `std::array` specializations.
 */

#include <array>
#include <cstddef>
#include <type_traits>

namespace gendil {

/** Identifies an exact `std::array<T, N>` specialization. */
template < class T >
struct is_std_array : std::false_type { };

template < class T, std::size_t N >
struct is_std_array< std::array< T, N > > : std::true_type { };

/**
 * Reports whether `T`, after removing cv- and reference qualifiers, is a
 * `std::array` specialization.
 */
template < class T >
inline constexpr bool is_std_array_v =
   is_std_array< std::remove_cvref_t< T > >::value;

} // namespace gendil
