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

} // namespace gendil
