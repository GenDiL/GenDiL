// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Meshes/Connectivities/Orientations/permutation.hpp"

namespace gendil {

/**
 * @brief Statically encoded identity orientation.
 *
 * Static identity is used by Cartesian connectivities to remove all runtime
 * orientation storage and arithmetic from generated kernels.
 */
template < Integer Dim >
using IdentityOrientation =
   std::integral_constant< Permutation< Dim >, MakeReferencePermutation< Dim >() >;

template < size_t Rank, Permutation< Rank > Value >
struct OrientationDimension<
   std::integral_constant< Permutation< Rank >, Value > >
   : std::integral_constant< size_t, Rank >
{};

template < class Orientation >
struct IsStaticIdentityOrientation : std::false_type
{};

template < size_t Rank >
struct IsStaticIdentityOrientation< IdentityOrientation< Rank > >
   : std::true_type
{};

template < class Orientation >
inline constexpr bool is_static_identity_orientation_v =
   IsStaticIdentityOrientation< std::remove_cvref_t< Orientation > >::value;

/** @brief Static identity orientations are valid by construction. */
template < size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool IsValidOrientation( const IdentityOrientation< Rank > & )
{
   return true;
}

/** @brief Materialize the flat identity permutation encoded by the type. */
template < size_t Rank, Permutation< Rank > Value >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr Permutation< Rank > FlattenOrientation(
   const std::integral_constant< Permutation< Rank >, Value > & )
{
   return Value;
}

} // namespace gendil
