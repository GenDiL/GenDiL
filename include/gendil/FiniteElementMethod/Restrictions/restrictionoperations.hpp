// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"

#include <type_traits>

namespace gendil {

/** Gather one restriction row, including all scalar-linear entry weights. */
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
            local_value = ApplyRestrictionWeight(
               weight,
               global_values[global_dof] );
         }
         else
         {
            local_value += ApplyRestrictionWeight(
               weight,
               global_values[global_dof] );
         }
      } );
   return local_value;
}

/** Assign the adjoint-weighted value to every visited row coordinate. */
template <
   typename Restriction,
   typename LocalDofIndex,
   typename LocalValue,
   typename GlobalValues >
   requires ElementDoFRestrictionFor< Restriction, LocalDofIndex >
GENDIL_HOST_DEVICE
constexpr void ScatterAssignRestrictionRowAdjoint(
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
         global_values[global_dof] = ApplyAdjointRestrictionWeight(
            weight,
            local_value );
      } );
}

/** Add the adjoint-weighted value to every visited row coordinate. */
template <
   typename Restriction,
   typename LocalDofIndex,
   typename LocalValue,
   typename GlobalValues >
   requires ElementDoFRestrictionFor< Restriction, LocalDofIndex >
GENDIL_HOST_DEVICE
constexpr void ScatterAddRestrictionRowAdjoint(
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
      [&] ( const GlobalIndex global_dof, const auto & weight )
      {
         global_values[global_dof] += ApplyAdjointRestrictionWeight(
            weight,
            local_value );
      } );
}

/** Atomically add the adjoint-weighted value to every visited coordinate. */
template <
   typename Restriction,
   typename LocalDofIndex,
   typename LocalValue,
   typename GlobalValues >
   requires ElementDoFRestrictionFor< Restriction, LocalDofIndex >
GENDIL_HOST_DEVICE
constexpr void AtomicScatterAddRestrictionRowAdjoint(
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
      [&] ( const GlobalIndex global_dof, const auto & weight )
      {
         AtomicAddInPlace(
            global_values[global_dof],
            ApplyAdjointRestrictionWeight( weight, local_value ) );
      } );
}

/** Compatibility name for additive adjoint scatter. */
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
   ScatterAddRestrictionRowAdjoint(
      restriction,
      element_index,
      local_dof,
      local_value,
      global_values );
}

} // namespace gendil
