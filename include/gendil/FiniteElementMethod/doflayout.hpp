// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementorders.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/vectorshapefunctions.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/multiindex.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

namespace gendil {

template < typename ShapeFunctions >
struct FiniteElementDofShape
{
   using type = orders_to_num_dofs< typename ShapeFunctions::orders >;
};

template < typename ... ScalarShapeFunctions >
struct FiniteElementDofShape< VectorShapeFunctions< ScalarShapeFunctions... > >
{
   using type = typename VectorShapeFunctions< ScalarShapeFunctions... >::dof_shape;
};

template < typename ShapeFunctions >
using finite_element_dof_shape_t =
   typename FiniteElementDofShape< ShapeFunctions >::type;

template <
   typename ShapeFunctions,
   size_t Component,
   bool IsVector = is_vector_shape_functions_v< ShapeFunctions > >
struct ComponentDofShape;

template < typename ShapeFunctions, size_t Component >
struct ComponentDofShape< ShapeFunctions, Component, false >
{
   static_assert(Component == 0, "Scalar finite element spaces only have component 0.");
   using type = finite_element_dof_shape_t< ShapeFunctions >;
};

template < typename ShapeFunctions, size_t Component >
struct ComponentDofShape< ShapeFunctions, Component, true >
{
   static_assert(Component < ShapeFunctions::vector_dim, "Vector component index is out of bounds.");
   using type = std::tuple_element_t< Component, typename ShapeFunctions::dof_shape >;
};

template < typename ShapeFunctions, size_t Component >
using component_dof_shape_t =
   typename ComponentDofShape< ShapeFunctions, Component >::type;

template < typename DofShapes, size_t ... I >
GENDIL_HOST_DEVICE
constexpr LocalIndex DofShapeTupleDofCount( std::index_sequence< I... > )
{
   return static_cast< LocalIndex >(
      ( 0 + ... + Product( std::tuple_element_t< I, DofShapes >{} ) ) );
}

template < typename ShapeFunctions >
GENDIL_HOST_DEVICE
constexpr LocalIndex LocalDofCount()
{
   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      return DofShapeTupleDofCount< typename ShapeFunctions::dof_shape >(
         std::make_index_sequence< ShapeFunctions::vector_dim >{} );
   }
   else
   {
      return static_cast< LocalIndex >(
         Product( finite_element_dof_shape_t< ShapeFunctions >{} ) );
   }
}

template < typename FESpace >
GENDIL_HOST_DEVICE
constexpr LocalIndex LocalDofCount( const FESpace & )
{
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   return LocalDofCount< ShapeFunctions >();
}

template < typename ShapeFunctions, size_t Component >
GENDIL_HOST_DEVICE
constexpr LocalIndex ComponentLocalDofOffset(
   std::integral_constant< size_t, Component > )
{
   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      return DofShapeTupleDofCount< typename ShapeFunctions::dof_shape >(
         std::make_index_sequence< Component >{} );
   }
   else
   {
      static_assert(Component == 0, "Scalar finite element spaces only have component 0.");
      return 0;
   }
}

template <
   typename ShapeFunctions,
   size_t Component,
   Integer Dim >
GENDIL_HOST_DEVICE
constexpr LocalIndex FlattenLocalDof(
   std::integral_constant< size_t, Component > component,
   const std::array< GlobalIndex, Dim > & indices )
{
   using DofShape = component_dof_shape_t< ShapeFunctions, Component >;
   // Local BSR block numbering is element-local and component-major for vector
   // spaces. It is intentionally separate from external FE-vector numbering.
   return ComponentLocalDofOffset< ShapeFunctions >( component ) +
      static_cast< LocalIndex >( FlattenMultiIndex< DofShape >( indices ) );
}

template < typename ShapeFunctions, Integer Dim >
GENDIL_HOST_DEVICE
constexpr LocalIndex FlattenLocalDof(
   const std::array< GlobalIndex, Dim > & indices )
{
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "Vector local DoF flattening requires a compile-time component tag." );
   return FlattenLocalDof< ShapeFunctions >(
      std::integral_constant< size_t, 0 >{},
      indices );
}

template <
   typename FESpace,
   size_t Component,
   Integer Dim >
GENDIL_HOST_DEVICE
constexpr LocalIndex FlattenLocalDof(
   const FESpace &,
   std::integral_constant< size_t, Component > component,
   const std::array< GlobalIndex, Dim > & indices )
{
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   return FlattenLocalDof< ShapeFunctions >( component, indices );
}

template < typename FESpace, Integer Dim >
GENDIL_HOST_DEVICE
constexpr LocalIndex FlattenLocalDof(
   const FESpace &,
   const std::array< GlobalIndex, Dim > & indices )
{
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   return FlattenLocalDof< ShapeFunctions >( indices );
}

template <
   typename DofShapes,
   size_t ... I >
GENDIL_HOST_DEVICE
size_t VectorOffset(
   DofShapes,
   GlobalIndex num_elements,
   std::index_sequence< I ... > )
{
   // Source of truth for component-major vector E-vector component offsets.
   return ( size_t{0} + ... +
      ( num_elements * Product( std::tuple_element_t< I, DofShapes >{} ) ) );
}

template <
   typename FESpace,
   size_t Component,
   Integer Dim >
GENDIL_HOST_DEVICE
GlobalIndex GlobalDofIndex(
   const FESpace & fe_space,
   std::integral_constant< size_t, Component > component,
   const GlobalIndex element_index,
   const std::array< GlobalIndex, Dim > & indices )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;
   static_assert(
      std::is_same_v< typename Space::restriction_type, L2Restriction >,
      "GlobalDofIndex currently supports L2Restriction finite element spaces." );

   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      using DofShapes = typename ShapeFunctions::dof_shape;
      using ComponentDofShape = component_dof_shape_t< ShapeFunctions, Component >;
      const GlobalIndex component_global_offset =
         VectorOffset(
            DofShapes{},
            fe_space.GetNumberOfFiniteElements(),
            std::make_index_sequence< Component >{} );
      const GlobalIndex component_dofs = Product( ComponentDofShape{} );
      return fe_space.restriction.shift +
         component_global_offset +
         element_index * component_dofs +
         static_cast< GlobalIndex >(
            FlattenMultiIndex< ComponentDofShape >( indices ) );
   }
   else
   {
      static_assert(Component == 0, "Scalar finite element spaces only have component 0.");
      using DofShape = finite_element_dof_shape_t< ShapeFunctions >;
      const GlobalIndex element_dofs = Product( DofShape{} );
      return fe_space.restriction.shift +
         element_index * element_dofs +
         static_cast< GlobalIndex >( FlattenMultiIndex< DofShape >( indices ) );
   }
}

template < typename FESpace, Integer Dim >
GENDIL_HOST_DEVICE
GlobalIndex GlobalDofIndex(
   const FESpace & fe_space,
   const GlobalIndex element_index,
   const std::array< GlobalIndex, Dim > & indices )
{
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "Vector global DoF indexing requires a compile-time component tag." );
   return GlobalDofIndex(
      fe_space,
      std::integral_constant< size_t, 0 >{},
      element_index,
      indices );
}

template <
   typename FESpace,
   size_t Component,
   Integer Dim >
GENDIL_HOST_DEVICE
GlobalIndex ElementToGlobalDofIndex(
   const FESpace & fe_space,
   std::integral_constant< size_t, Component > component,
   const GlobalIndex element_index,
   const std::array< GlobalIndex, Dim > & indices )
{
   using Space = std::remove_cvref_t< FESpace >;
   using Restriction = typename Space::restriction_type;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;

   if constexpr ( std::is_same_v< Restriction, L2Restriction > )
   {
      return GlobalDofIndex( fe_space, component, element_index, indices );
   }
   else if constexpr ( std::is_same_v< Restriction, H1Restriction > )
   {
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "ElementToGlobalDofIndex currently supports scalar H1 spaces only; vector H1 is not supported." );
      static_assert(Component == 0, "Scalar H1 finite element spaces only have component 0.");

      constexpr LocalIndex local_dofs = LocalDofCount< ShapeFunctions >();
      const LocalIndex local_id = FlattenLocalDof( fe_space, component, indices );
      const GlobalIndex restriction_index =
         element_index * static_cast< GlobalIndex >( local_dofs ) +
         static_cast< GlobalIndex >( local_id );
      const int global_index = fe_space.restriction.indices[restriction_index];
      GENDIL_VERIFY(
         global_index >= 0,
         "H1Restriction contains a negative element-to-global DoF index." );
      return static_cast< GlobalIndex >( global_index );
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "ElementToGlobalDofIndex supports only L2Restriction and scalar H1Restriction." );
      return 0;
   }
}

template < typename FESpace, Integer Dim >
GENDIL_HOST_DEVICE
GlobalIndex ElementToGlobalDofIndex(
   const FESpace & fe_space,
   const GlobalIndex element_index,
   const std::array< GlobalIndex, Dim > & indices )
{
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "Vector element-to-global DoF indexing requires a compile-time component tag." );
   return ElementToGlobalDofIndex(
      fe_space,
      std::integral_constant< size_t, 0 >{},
      element_index,
      indices );
}

} // namespace gendil
