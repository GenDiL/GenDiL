// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Checked arithmetic for unsigned counts, sizes, and offsets.
 *
 * Host execution rejects overflow, including release and constant evaluation.
 * Device compilation emits only arithmetic; callers must establish
 * representability through host setup or static type checks.
 */

#include "gendil/Utilities/debug.hpp"

#include <concepts>
#include <limits>
#include <type_traits>

namespace gendil {

/** @brief Add two unsigned integers after checking for overflow. */
template < typename T >
   requires (
      std::unsigned_integral< T > &&
      !std::same_as< std::remove_cv_t< T >, bool > )
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr T CheckedAdd(
   const T lhs,
   const T rhs,
   const char * message )
{
   GENDIL_VERIFY(
      rhs <= std::numeric_limits< T >::max() - lhs,
      message );
   return lhs + rhs;
}

/** @brief Multiply two unsigned integers after checking for overflow. */
template < typename T >
   requires (
      std::unsigned_integral< T > &&
      !std::same_as< std::remove_cv_t< T >, bool > )
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr T CheckedMultiply(
   const T lhs,
   const T rhs,
   const char * message )
{
   GENDIL_VERIFY(
      lhs == 0 || rhs <= std::numeric_limits< T >::max() / lhs,
      message );
   return lhs * rhs;
}

} // namespace gendil
