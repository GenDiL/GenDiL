// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

#include <array>
#include <concepts>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Require a compile-time element-local DoF tensor shape.
 *
 * A static DoF shape is default-constructible metadata whose @c size()
 * reports its tensor rank and whose extents can be multiplied by @c Product.
 */
template < typename Shape >
concept StaticDofShape = requires
{
   { Shape::size() } -> std::convertible_to< size_t >;
   Product( Shape{} );
};

/**
 * @brief Require a completed restriction with one tensor-shaped local DoF
 * index space.
 *
 * The restriction exposes a @c dof_shape_type and a matching @c tensor_dim,
 * final local/global dimensions, an algebraic DoF extent, and row
 * visitation for a native @c std::array local DoF coordinate. The visitor is
 * invoked once for every nonzero entry in the selected restriction row.
 */
template < typename Restriction >
concept TensorElementDoFRestriction =
   requires
   {
      typename std::remove_cvref_t< Restriction >::dof_shape_type;
      std::remove_cvref_t< Restriction >::tensor_dim;
   } &&
   StaticDofShape<
      typename std::remove_cvref_t< Restriction >::dof_shape_type > &&
   std::integral< std::remove_cv_t< decltype(
      std::remove_cvref_t< Restriction >::tensor_dim ) > > &&
   ( std::remove_cvref_t< Restriction >::tensor_dim ==
     static_cast< Integer >(
        std::remove_cvref_t< Restriction >::dof_shape_type::size() ) ) &&
   requires(
      const std::remove_cvref_t< Restriction > & restriction,
      const GlobalIndex element,
      const std::array<
         GlobalIndex,
         std::remove_cvref_t< Restriction >::tensor_dim > & local_dof )
   {
      { GetNumberOfLocalDofs( restriction ) }
         -> std::same_as< GlobalIndex >;
      { GetNumberOfGlobalDofs( restriction ) }
         -> std::same_as< GlobalIndex >;
      { GetAlgebraicDofExtent( restriction ) }
         -> std::same_as< GlobalIndex >;
      ForEachRestrictionEntry(
         restriction,
         element,
         local_dof,
         []( GlobalIndex, const auto & ) { } );
   };

/**
 * @brief Require one compile-time component of a vector restriction.
 *
 * Component @p Component must expose a tensor restriction through
 * @c GetComponentRestriction<Component> and accept the corresponding
 * @c LocalComponentDoFIndex through wrapper-level row visitation.
 */
template < typename Restriction, size_t Component >
concept VectorRestrictionComponent =
   requires( const std::remove_cvref_t< Restriction > & restriction )
   {
      GetComponentRestriction< Component >( restriction );
   } &&
   TensorElementDoFRestriction< std::remove_cvref_t< decltype(
      GetComponentRestriction< Component >(
         std::declval<
            const std::remove_cvref_t< Restriction > & >() ) ) > > &&
   requires(
      const std::remove_cvref_t< Restriction > & restriction,
      const GlobalIndex element,
      const LocalComponentDoFIndex<
         Component,
         std::remove_cvref_t< decltype(
            GetComponentRestriction< Component >(
               std::declval<
                  const std::remove_cvref_t< Restriction > & >() ) )
            >::tensor_dim > & local_dof )
   {
      ForEachRestrictionEntry(
         restriction,
         element,
         local_dof,
         []( GlobalIndex, const auto & ) { } );
   };

/**
 * @brief Require a completed compile-time aggregation of tensor restrictions.
 *
 * A vector restriction declares a positive @c num_components, exposes final
 * aggregate dimensions, and provides @c GetComponentRestriction<C> for every
 * component. Each selected component is a @c TensorElementDoFRestriction and
 * can be visited through the corresponding @c LocalComponentDoFIndex.
 */
template < typename Restriction >
concept VectorElementDoFRestriction =
   requires( const std::remove_cvref_t< Restriction > & restriction )
   {
      std::remove_cvref_t< Restriction >::num_components;
      { GetNumberOfLocalDofs( restriction ) }
         -> std::same_as< GlobalIndex >;
      { GetNumberOfGlobalDofs( restriction ) }
         -> std::same_as< GlobalIndex >;
      { GetAlgebraicDofExtent( restriction ) }
         -> std::same_as< GlobalIndex >;
   } &&
   std::integral< std::remove_cv_t< decltype(
      std::remove_cvref_t< Restriction >::num_components ) > > &&
   ( std::remove_cvref_t< Restriction >::num_components > 0 ) &&
   []< size_t... Component >(
      std::index_sequence< Component... > ) consteval
   {
      return
         ( VectorRestrictionComponent< Restriction, Component > && ... );
   }( std::make_index_sequence<
      std::remove_cvref_t< Restriction >::num_components >{} );

/**
 * @brief Require any completed element-to-global DoF restriction.
 *
 * This umbrella accepts either a tensor-shaped leaf restriction or a
 * compile-time vector restriction. Use the refined concepts when code depends
 * on one of their distinct local-index contracts.
 */
template < typename Restriction >
concept ElementDoFRestriction =
   TensorElementDoFRestriction< Restriction > ||
   VectorElementDoFRestriction< Restriction >;

/**
 * @brief Require independent element rows backed by unit-weight references.
 *
 * This derived capability is intended for elementwise traversals that may
 * overwrite their destination. It does not by itself describe a general
 * parallel scatter policy.
 */
template < typename Restriction >
concept ElementwiseIndependentRestriction =
   ElementDoFRestriction< Restriction > &&
   static_restriction_entry_count_v<
      std::remove_cvref_t< Restriction > > == 1 &&
   restriction_supports_element_reference_view_v< Restriction > &&
   !restriction_may_share_global_dofs_v<
      std::remove_cvref_t< Restriction > >;

/**
 * @brief Require a contextual restriction specification for a mesh and finite
 * element.
 *
 * A specification is recognized through an unqualified, ADL-customizable
 * @c MakeElementDoFRestriction(mesh, finite_element, specification) call that
 * returns a completed @c ElementDoFRestriction. Completed restrictions are
 * explicitly excluded from this construction path.
 */
template < typename Specification, typename Mesh, typename FiniteElement >
concept RestrictionSpecificationFor =
   !ElementDoFRestriction< std::remove_cvref_t< Specification > > &&
   requires(
      const Mesh & mesh,
      const FiniteElement & finite_element,
      const Specification & specification )
   {
      { MakeElementDoFRestriction(
         mesh,
         finite_element,
         specification ) } -> ElementDoFRestriction;
   };

/**
 * @brief Require a completed restriction to accept a particular local DoF
 * index type.
 *
 * This contextual refinement checks that @c ForEachRestrictionEntry is a
 * valid expression for @p LocalDofIndex in addition to requiring the completed
 * restriction contract.
 */
template < typename Restriction, typename LocalDofIndex >
concept ElementDoFRestrictionFor =
   ElementDoFRestriction< Restriction > &&
   requires(
      const std::remove_cvref_t< Restriction > & restriction,
      const GlobalIndex element,
      const LocalDofIndex & local_dof )
   {
      ForEachRestrictionEntry(
         restriction,
         element,
         local_dof,
         []( GlobalIndex, const auto & ) { } );
   };

/**
 * @brief Require a tensor restriction whose local DoF shape matches scalar
 * shape functions.
 */
template < typename Restriction, typename ShapeFunctions >
concept TensorElementDoFRestrictionForShapeFunctions =
   !is_vector_shape_functions_v< ShapeFunctions > &&
   TensorElementDoFRestriction< Restriction > &&
   std::same_as<
      typename std::remove_cvref_t< Restriction >::dof_shape_type,
      finite_element_dof_shape_t< ShapeFunctions > >;

/**
 * @brief Require a vector restriction whose component shapes match vector
 * shape functions.
 */
template < typename Restriction, typename ShapeFunctions >
concept VectorElementDoFRestrictionForShapeFunctions =
   is_vector_shape_functions_v< ShapeFunctions > &&
   VectorElementDoFRestriction< Restriction > &&
   ( std::remove_cvref_t< Restriction >::num_components ==
     ShapeFunctions::vector_dim ) &&
   []< size_t... Component >(
      std::index_sequence< Component... > ) consteval
   {
      return
         ( std::same_as<
              typename std::remove_cvref_t< decltype(
                 GetComponentRestriction< Component >(
                    std::declval<
                       const std::remove_cvref_t< Restriction > & >() ) )
                 >::dof_shape_type,
              component_dof_shape_t< ShapeFunctions, Component > > && ... );
   }( std::make_index_sequence< ShapeFunctions::vector_dim >{} );

/**
 * @brief Require a completed restriction whose native local DoF structure
 * matches a finite element.
 *
 * This is a compile-time shape and component compatibility check. Runtime
 * compatibility with a particular mesh, including the total local row count,
 * is checked by @c ValidateElementDoFRestrictionFor.
 */
template < typename Restriction, typename FiniteElement >
concept CompatibleElementDoFRestrictionFor =
   requires
   {
      typename std::remove_cvref_t< FiniteElement >::shape_functions;
   } &&
   (
      TensorElementDoFRestrictionForShapeFunctions<
         Restriction,
         typename std::remove_cvref_t<
            FiniteElement >::shape_functions > ||
      VectorElementDoFRestrictionForShapeFunctions<
         Restriction,
         typename std::remove_cvref_t<
            FiniteElement >::shape_functions >
   );

} // namespace gendil
