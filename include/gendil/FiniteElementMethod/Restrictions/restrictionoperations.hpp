// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"

#include <type_traits>

namespace gendil {

namespace details {

template < typename Value >
GENDIL_HOST_DEVICE
constexpr const Value & ApplyRestrictionEntryWeight(
   RestrictionUnitWeight,
   const Value & value )
{
   return value;
}

template < typename Value >
GENDIL_HOST_DEVICE
constexpr const Value & ApplyAdjointRestrictionEntryWeight(
   RestrictionUnitWeight,
   const Value & value )
{
   return value;
}

} // namespace details

/** Reference gather for the currently implemented unit-weight row entries. */
template <
   typename Restriction,
   typename LocalDofIndex,
   typename GlobalValues >
   requires ElementDoFRestrictionFor< Restriction, LocalDofIndex >
GENDIL_HOST_DEVICE
constexpr auto GatherRestrictionRow(
   const Restriction & restriction,
   const GlobalIndex element_index,
   const LocalDofIndex & local_dof,
   const GlobalValues & global_values )
{
   using Value = std::remove_cvref_t< decltype( global_values[0] ) >;
   Value local_value{};
   ForEachRestrictionEntry(
      restriction,
      element_index,
      local_dof,
      [&] (
         const GlobalIndex global_dof,
         const auto & weight )
      {
         if constexpr (
            static_restriction_entry_count_v<
               std::remove_cvref_t< Restriction > > == 1 )
         {
            local_value = details::ApplyRestrictionEntryWeight(
               weight,
               global_values[global_dof] );
         }
         else
         {
            local_value += details::ApplyRestrictionEntryWeight(
               weight,
               global_values[global_dof] );
         }
      } );
   return local_value;
}

/** Reference adjoint scatter for the currently implemented unit-weight rows. */
template <
   typename Restriction,
   typename LocalDofIndex,
   typename LocalValue,
   typename GlobalValues >
   requires ElementDoFRestrictionFor< Restriction, LocalDofIndex >
GENDIL_HOST_DEVICE
constexpr void ScatterRestrictionRowAdjoint(
   const Restriction & restriction,
   const GlobalIndex element_index,
   const LocalDofIndex & local_dof,
   const LocalValue & local_value,
   GlobalValues & global_values )
{
   ForEachRestrictionEntry(
      restriction,
      element_index,
      local_dof,
      [&] (
         const GlobalIndex global_dof,
         const auto & weight )
      {
         global_values[global_dof] +=
            details::ApplyAdjointRestrictionEntryWeight(
               weight,
               local_value );
      } );
}

} // namespace gendil
