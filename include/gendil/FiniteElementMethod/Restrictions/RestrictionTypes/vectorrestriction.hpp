// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Compile-time aggregation of component element-DoF restrictions.
 */

#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionvalidation.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Direct sum of statically selected component restrictions.
 *
 * Every child emits coordinates in the final vector-wide algebraic space.
 * Component selection is encoded in @c LocalComponentDoFIndex and therefore
 * introduces no runtime dispatch.
 */
template < typename... ComponentRestrictions >
class VectorRestriction
{
public:
   static_assert(
      sizeof...( ComponentRestrictions ) > 0,
      "VectorRestriction requires at least one component restriction." );

   static constexpr Integer num_components =
      sizeof...( ComponentRestrictions );

   using component_restrictions_type =
      std::tuple< ComponentRestrictions... >;

   component_restrictions_type component_restrictions;
   GlobalIndex num_local_dofs;
   GlobalIndex num_global_dofs;
   GlobalIndex algebraic_dof_extent;
};

/** @brief Semantic name for a vector restriction composed of L2 leaves. */
template < typename... ComponentRestrictions >
using VectorL2Restriction =
   VectorRestriction< ComponentRestrictions... >;

/** @brief Return a statically selected component restriction. */
template < size_t Component, typename... ComponentRestrictions >
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetComponentRestriction(
   const VectorRestriction< ComponentRestrictions... > & restriction )
{
   static_assert(
      Component < sizeof...( ComponentRestrictions ),
      "Vector restriction component index is out of bounds." );
   return std::get< Component >( restriction.component_restrictions );
}

/** @brief Visit a vector row by delegating to its selected component. */
template <
   size_t Component,
   Integer Dim,
   typename... ComponentRestrictions,
   typename Visitor >
GENDIL_HOST_DEVICE
constexpr void ForEachRestrictionEntry(
   const VectorRestriction< ComponentRestrictions... > & restriction,
   const GlobalIndex element,
   const LocalComponentDoFIndex< Component, Dim > & local_dof,
   Visitor && visitor )
{
   const auto & child = GetComponentRestriction< Component >( restriction );
   using Child = std::remove_cvref_t< decltype( child ) >;
   static_assert(
      Dim == Child::tensor_dim,
      "Vector component tensor rank does not match its restriction." );
   ForEachRestrictionEntry(
      child,
      element,
      local_dof.local_dof,
      std::forward< Visitor >( visitor ) );
}

/** @brief Return the sum of the component-local restriction row counts. */
template < typename... ComponentRestrictions >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfLocalDofs(
   const VectorRestriction< ComponentRestrictions... > & restriction )
{
   return restriction.num_local_dofs;
}

/** @brief Return the sum of the component logical global DoF counts. */
template < typename... ComponentRestrictions >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfGlobalDofs(
   const VectorRestriction< ComponentRestrictions... > & restriction )
{
   return restriction.num_global_dofs;
}

/** @brief Return the shared vector-wide algebraic coordinate extent. */
template < typename... ComponentRestrictions >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetAlgebraicDofExtent(
   const VectorRestriction< ComponentRestrictions... > & restriction )
{
   return restriction.algebraic_dof_extent;
}

template < typename... ComponentRestrictions >
inline constexpr size_t static_restriction_entry_count_v<
   VectorRestriction< ComponentRestrictions... > > = []
{
   if constexpr (
      ( ( static_restriction_entry_count_v< ComponentRestrictions > !=
            dynamic_restriction_entry_count ) && ... ) &&
      ( ( static_restriction_entry_count_v< ComponentRestrictions > ==
            static_restriction_entry_count_v<
               std::tuple_element_t<
                  0,
                  std::tuple< ComponentRestrictions... > > > ) && ... ) )
   {
      return static_restriction_entry_count_v<
         std::tuple_element_t<
            0,
            std::tuple< ComponentRestrictions... > > >;
   }
   else
   {
      return dynamic_restriction_entry_count;
   }
}();

template < typename... ComponentRestrictions >
struct restriction_supports_element_reference_view<
   VectorRestriction< ComponentRestrictions... > >
   : std::bool_constant<
        ( restriction_supports_element_reference_view_v<
             ComponentRestrictions > && ... ) >
{ };

/** @brief Validate memory access recursively for every component map. */
template <
   bool OnDevice,
   typename... ComponentRestrictions >
void ValidateRestrictionMemoryAccess(
   const VectorRestriction< ComponentRestrictions... > & restriction,
   const GlobalIndex active_row_count )
{
   std::apply(
      [&] ( const auto & ... component )
      {
         ( ValidateRestrictionMemoryAccess< OnDevice >(
              component,
              active_row_count ),
           ... );
      },
      restriction.component_restrictions );
}

namespace details {

template < typename Restriction, size_t... Component >
void ValidateVectorElementDoFRestriction(
   const Restriction & restriction,
   std::index_sequence< Component... > )
{
   GlobalIndex local_sum = 0;
   GlobalIndex global_sum = 0;
   auto validate_component = [&]< size_t C >(
      std::integral_constant< size_t, C > )
   {
      const auto & component = GetComponentRestriction< C >( restriction );
      ValidateElementDoFRestriction( component );
      GENDIL_VERIFY(
         GetAlgebraicDofExtent( component ) ==
            restriction.algebraic_dof_extent,
         "Every vector component restriction must report the same final algebraic extent." );
      local_sum = CheckedAdd(
         local_sum,
         GetNumberOfLocalDofs( component ),
         "Vector restriction local extent overflow." );
      global_sum = CheckedAdd(
         global_sum,
         GetNumberOfGlobalDofs( component ),
         "Vector restriction logical global DoF count overflow." );
   };
   ( validate_component(
        std::integral_constant< size_t, Component >{} ),
     ... );
   GENDIL_VERIFY(
      local_sum == restriction.num_local_dofs,
      "Vector restriction local extent is not the sum of its component extents." );
   GENDIL_VERIFY(
      global_sum == restriction.num_global_dofs,
      "Vector restriction global DoF count is not the sum of its component counts." );
   GENDIL_VERIFY(
      restriction.num_global_dofs <= restriction.algebraic_dof_extent,
      "Vector restriction global DoF count exceeds its algebraic extent." );

}

} // namespace details

/** @brief Validate component dimensions, extents, and direct-sum placement. */
template < typename... ComponentRestrictions >
void ValidateRestrictionRepresentation(
   const VectorRestriction< ComponentRestrictions... > & restriction )
{
   details::ValidateVectorElementDoFRestriction(
      restriction,
      std::index_sequence_for< ComponentRestrictions... >{} );
}

} // namespace gendil
