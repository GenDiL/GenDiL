// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Default tensor layout backed by a completed element-DoF restriction.
 */

#include "gendil/FiniteElementMethod/Restrictions/globaldofindex.hpp"

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Adapt a statically one-entry tensor restriction to the View layout
 * interface.
 *
 * The final layout coordinate is the element ordinal. All preceding
 * coordinates form the restriction's native local tensor index.
 */
template < typename Restriction >
   requires (
      TensorElementDoFRestriction< Restriction > &&
      static_restriction_entry_count_v<
         std::remove_cvref_t< Restriction > > == 1 )
struct RestrictionLayout
{
   using restriction_type = std::remove_cvref_t< Restriction >;

   static constexpr size_t rank =
      static_cast< size_t >( restriction_type::tensor_dim ) + 1;

   restriction_type restriction;

private:
   template < typename Indices, size_t... LocalIndex >
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Offset(
      const Indices & indices,
      std::index_sequence< LocalIndex... > ) const
   {
      const std::array< GlobalIndex, sizeof...( LocalIndex ) > local_dof{
         static_cast< GlobalIndex >(
            std::get< LocalIndex >( indices ) )... };
      return GetGlobalDofIndex(
         restriction,
         static_cast< GlobalIndex >( std::get< rank - 1 >( indices ) ),
         local_dof );
   }

public:
   template < typename... Indices >
      requires ( sizeof...( Indices ) == rank )
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Offset( Indices... indices ) const
   {
      const auto coordinate = std::make_tuple( indices... );
      return Offset(
         coordinate,
         std::make_index_sequence< rank - 1 >{} );
   }
};

/**
 * @brief Construct the default semantic layout for a tensor restriction.
 *
 * This free function is intentionally an ADL customization point. A
 * specialized overload must produce the same offsets as @c GetGlobalDofIndex
 * for every valid restriction row.
 */
template < typename Restriction >
   requires (
      TensorElementDoFRestriction< Restriction > &&
      static_restriction_entry_count_v<
         std::remove_cvref_t< Restriction > > == 1 )
GENDIL_HOST_DEVICE
constexpr auto MakeRestrictionLayout( const Restriction & restriction )
{
   using RestrictionType = std::remove_cvref_t< Restriction >;
   return RestrictionLayout< RestrictionType >{ restriction };
}

} // namespace gendil
