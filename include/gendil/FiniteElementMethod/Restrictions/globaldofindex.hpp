// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Singular global DoF lookup for restrictions and finite-element spaces.
 *
 * @c GetGlobalDofIndex is available only when a restriction row has exactly
 * one statically known entry. General zero-, multi-entry, or dynamically sized
 * rows must use @c ForEachRestrictionEntry.
 */

#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofdescriptor.hpp"

#include <array>
#include <type_traits>

namespace gendil {

/**
 * @brief Return the global coordinate of a statically one-entry restriction
 * row.
 *
 * This coordinate-only convenience operation intentionally discards the
 * entry weight. Use @c ForEachRestrictionEntry when the weight is required.
 */
template < typename Restriction, typename LocalDofIndex >
   requires (
      ElementDoFRestrictionFor< Restriction, LocalDofIndex > &&
      static_restriction_entry_count_v<
         std::remove_cvref_t< Restriction > > == 1 )
GENDIL_HOST_DEVICE
GlobalIndex GetGlobalDofIndex(
   const Restriction & restriction,
   const GlobalIndex element_index,
   const LocalDofIndex & local_dof )
{
   GlobalIndex algebraic_dof = 0;
   ForEachRestrictionEntry(
      restriction,
      element_index,
      local_dof,
      [&] ( const GlobalIndex index, const auto & )
      {
         algebraic_dof = index;
      } );
   return algebraic_dof;
}

/** @brief Look up a scalar space DoF from a flattened local ordinal. */
template < typename FESpace >
   requires (
      !ElementDoFRestriction< std::remove_cvref_t< FESpace > > &&
      !is_vector_finite_element_space_v< FESpace > )
GENDIL_HOST_DEVICE
GlobalIndex GetGlobalDofIndex(
   const FESpace & fe_space,
   const GlobalIndex element_index,
   const GlobalIndex scalar_local_dof_index )
{
   using ShapeFunctions =
      finite_element_space_shape_functions_t< FESpace >;
   using DofShape = finite_element_dof_shape_t< ShapeFunctions >;
   return GetGlobalDofIndex(
      fe_space.restriction,
      element_index,
      UnflattenMultiIndex< DofShape >( scalar_local_dof_index ) );
}

/**
 * @brief Look up a scalar DoF through the component-tagged traversal API.
 *
 * Generic local-DoF traversals carry a compile-time component tag for both
 * scalar and vector spaces. Component zero is the unique valid scalar tag.
 */
template < typename FESpace, size_t Component >
   requires (
      !ElementDoFRestriction< std::remove_cvref_t< FESpace > > &&
      !is_vector_finite_element_space_v< FESpace > &&
      Component == 0 )
GENDIL_HOST_DEVICE
GlobalIndex GetGlobalDofIndex(
   const FESpace & fe_space,
   std::integral_constant< size_t, Component >,
   const GlobalIndex element_index,
   const GlobalIndex scalar_local_dof_index )
{
   return GetGlobalDofIndex(
      fe_space,
      element_index,
      scalar_local_dof_index );
}

/**
 * @brief Look up a statically selected component from a flattened component
 * ordinal.
 */
template < typename FESpace, size_t Component >
   requires (
      !ElementDoFRestriction< std::remove_cvref_t< FESpace > > &&
      is_vector_finite_element_space_v< FESpace > )
GENDIL_HOST_DEVICE
GlobalIndex GetGlobalDofIndex(
   const FESpace & fe_space,
   std::integral_constant< size_t, Component >,
   const GlobalIndex element_index,
   const GlobalIndex component_local_dof_index )
{
   using ShapeFunctions =
      finite_element_space_shape_functions_t< FESpace >;
   using DofShape =
      component_dof_shape_t< ShapeFunctions, Component >;
   const auto local_dof = LocalComponentDoFIndex<
      Component,
      static_cast< Integer >( DofShape::size() ) >{
         UnflattenMultiIndex< DofShape >(
            component_local_dof_index ) };
   return GetGlobalDofIndex(
      fe_space.restriction,
      element_index,
      local_dof );
}

/**
 * @brief Look up a statically selected component from its tensor coordinate.
 */
template < typename FESpace, size_t Component, Integer Dim >
   requires (
      !ElementDoFRestriction< std::remove_cvref_t< FESpace > > &&
      is_vector_finite_element_space_v< FESpace > )
GENDIL_HOST_DEVICE
GlobalIndex GetGlobalDofIndex(
   const FESpace & fe_space,
   std::integral_constant< size_t, Component >,
   const GlobalIndex element_index,
   const std::array< GlobalIndex, Dim > & indices )
{
   return GetGlobalDofIndex(
      fe_space.restriction,
      element_index,
      LocalComponentDoFIndex< Component, Dim >{
         indices } );
}

/** @brief Look up a scalar space DoF from its tensor coordinate. */
template < typename FESpace, Integer Dim >
   requires (
      !ElementDoFRestriction< std::remove_cvref_t< FESpace > > &&
      !is_vector_finite_element_space_v< FESpace > )
GENDIL_HOST_DEVICE
GlobalIndex GetGlobalDofIndex(
   const FESpace & fe_space,
   const GlobalIndex element_index,
   const std::array< GlobalIndex, Dim > & indices )
{
   return GetGlobalDofIndex(
      fe_space.restriction,
      element_index,
      indices );
}

/** @brief Look up a scalar tensor coordinate tagged as component zero. */
template <
   typename FESpace,
   size_t Component,
   Integer Dim >
   requires (
      !ElementDoFRestriction< std::remove_cvref_t< FESpace > > &&
      !is_vector_finite_element_space_v< FESpace > &&
      Component == 0 )
GENDIL_HOST_DEVICE
GlobalIndex GetGlobalDofIndex(
   const FESpace & fe_space,
   std::integral_constant< size_t, Component >,
   const GlobalIndex element_index,
   const std::array< GlobalIndex, Dim > & indices )
{
   return GetGlobalDofIndex(
      fe_space,
      element_index,
      indices );
}

/**
 * @brief Look up a DoF from a matrix-free local descriptor.
 *
 * Scalar and vector descriptors delegate uniformly to the component-tagged
 * array API. A scalar descriptor must carry the canonical component-zero tag.
 */
template < typename FESpace, typename Descriptor >
   requires (
      !ElementDoFRestriction< std::remove_cvref_t< FESpace > > &&
      is_local_dof_descriptor_v< Descriptor > &&
      std::remove_cvref_t< Descriptor >::is_vector ==
         is_vector_finite_element_space_v< FESpace > &&
      ( std::remove_cvref_t< Descriptor >::is_vector ||
        std::remove_cvref_t< Descriptor >::component_id == 0 ) )
GENDIL_HOST_DEVICE
GlobalIndex GetGlobalDofIndex(
   const FESpace & fe_space,
   const GlobalIndex element_index,
   const Descriptor & local_dof )
{
   using DofDescriptor = std::remove_cvref_t< Descriptor >;
   return GetGlobalDofIndex(
      fe_space,
      typename DofDescriptor::component{},
      element_index,
      local_dof.indices );
}

} // namespace gendil
