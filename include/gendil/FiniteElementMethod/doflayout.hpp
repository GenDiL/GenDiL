// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementorders.hpp"
#include "gendil/FiniteElementMethod/restriction.hpp"
#include "gendil/FiniteElementMethod/tensorproductdoflayout.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/vectorshapefunctions.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/multiindex.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

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

template < typename ShapeFunctions, size_t ... I >
GENDIL_HOST_DEVICE
constexpr bool VectorComponentDofShapesMatchFirst_impl(
   std::index_sequence< I... > )
{
   using FirstDofShape = component_dof_shape_t< ShapeFunctions, 0 >;
   return ( std::is_same_v<
      FirstDofShape,
      component_dof_shape_t< ShapeFunctions, I > > && ... );
}

template < typename ShapeFunctions >
GENDIL_HOST_DEVICE
constexpr bool VectorComponentDofShapesMatchFirst()
{
   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      return VectorComponentDofShapesMatchFirst_impl< ShapeFunctions >(
         std::make_index_sequence< ShapeFunctions::vector_dim >{} );
   }
   else
   {
      return false;
   }
}

template < typename DofShapes, size_t ... I >
GENDIL_HOST_DEVICE
constexpr GlobalIndex DofShapeTupleDofCount( std::index_sequence< I... > )
{
   return ( GlobalIndex{0} + ... +
      Product( std::tuple_element_t< I, DofShapes >{} ) );
}

template < typename ShapeFunctions >
GENDIL_HOST_DEVICE
constexpr GlobalIndex LocalDofCount()
{
   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      return DofShapeTupleDofCount< typename ShapeFunctions::dof_shape >(
         std::make_index_sequence< ShapeFunctions::vector_dim >{} );
   }
   else
   {
      return Product( finite_element_dof_shape_t< ShapeFunctions >{} );
   }
}

template < typename FESpace >
GENDIL_HOST_DEVICE
constexpr GlobalIndex LocalDofCount( const FESpace & )
{
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   return LocalDofCount< ShapeFunctions >();
}

template < typename ShapeFunctions, size_t Component >
GENDIL_HOST_DEVICE
constexpr GlobalIndex ComponentLocalDofOffset(
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
constexpr GlobalIndex FlattenComponentLocalDof(
   std::integral_constant< size_t, Component >,
   const std::array< GlobalIndex, Dim > & indices )
{
   using DofShape = component_dof_shape_t< ShapeFunctions, Component >;
   return FlattenMultiIndex< DofShape >( indices );
}

template <
   typename ShapeFunctions,
   size_t Component,
   Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex FlattenLocalDof(
   std::integral_constant< size_t, Component > component,
   const std::array< GlobalIndex, Dim > & indices )
{
   // Local BSR block numbering is element-local and component-major for vector
   // spaces. It is intentionally separate from external FE-vector numbering.
   return ComponentLocalDofOffset< ShapeFunctions >( component ) +
      FlattenComponentLocalDof< ShapeFunctions >( component, indices );
}

template < typename ShapeFunctions, Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex FlattenLocalDof(
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
constexpr GlobalIndex FlattenLocalDof(
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
constexpr GlobalIndex FlattenLocalDof(
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

template < typename FESpace >
GENDIL_HOST_DEVICE
GlobalIndex ScalarLocalDofCount( const FESpace & )
{
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "ScalarLocalDofCount supports scalar finite element spaces only." );
   return LocalDofCount< ShapeFunctions >();
}

template < typename FESpace >
GENDIL_HOST_DEVICE
GlobalIndex ScalarGlobalTopologyDofCount( const FESpace & fe_space )
{
   using Space = std::remove_cvref_t< FESpace >;
   using Restriction = typename Space::restriction_type;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;

   if constexpr ( std::is_same_v< Restriction, L2Restriction > )
   {
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "ScalarGlobalTopologyDofCount supports scalar L2 finite element spaces only." );
      return fe_space.GetNumberOfFiniteElements() *
         ScalarLocalDofCount( fe_space );
   }
   else if constexpr ( std::is_same_v< Restriction, H1Restriction > )
   {
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "H1Restriction is scalar-only; use VectorH1Restriction<NComp> for vector H1 spaces." );
      return fe_space.restriction.num_dofs;
   }
   else if constexpr ( is_vector_h1_restriction_v< Restriction > )
   {
      static_assert(
         is_vector_shape_functions_v< ShapeFunctions >,
         "VectorH1Restriction requires a vector finite element space." );
      return fe_space.restriction.scalar_num_dofs;
   }
   else if constexpr ( is_tensor_product_restriction_v< Restriction > )
   {
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "TensorProductRestriction v1 supports scalar finite element spaces only." );
      return fe_space.restriction.num_dofs;
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "ScalarGlobalTopologyDofCount supports only scalar L2Restriction, scalar H1Restriction, VectorH1Restriction, and scalar TensorProductRestriction." );
      return 0;
   }
}

template < typename ... FactorSpaces >
auto MakeTensorProductRestriction( const FactorSpaces & ... factor_spaces )
{
   using Restriction = TensorProductRestriction<
      TensorProductRestrictionFactor<
         typename std::remove_cvref_t< FactorSpaces >::restriction_type,
         finite_element_dof_shape_t<
            typename std::remove_cvref_t<
               FactorSpaces >::finite_element_type::shape_functions > >... >;

   static_assert(
      ( !is_vector_shape_functions_v<
           typename std::remove_cvref_t<
              FactorSpaces >::finite_element_type::shape_functions > && ... ),
      "TensorProductRestriction v1 supports scalar factor spaces only." );

   const std::array< GlobalIndex, sizeof...( FactorSpaces ) > element_counts{
      static_cast< GlobalIndex >(
         factor_spaces.GetNumberOfFiniteElements() )... };
   const std::array< GlobalIndex, sizeof...( FactorSpaces ) > global_dof_counts{
      ScalarGlobalTopologyDofCount( factor_spaces )... };

   return Restriction{
      std::make_tuple( factor_spaces.restriction... ),
      MakePrefixStrides( element_counts ),
      MakePrefixStrides( global_dof_counts ),
      Product( global_dof_counts )
   };
}

template <
   typename FESpace >
GENDIL_HOST_DEVICE
Integer FiniteElementDofCount( const FESpace & fe_space )
{
   using Space = std::remove_cvref_t< FESpace >;
   using Restriction = typename Space::restriction_type;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;

   if constexpr ( std::is_same_v< Restriction, L2Restriction > )
   {
      if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
      {
         return fe_space.GetNumberOfFiniteElements() *
            static_cast< Integer >( LocalDofCount< ShapeFunctions >() );
      }
      else
      {
         return static_cast< Integer >(
            ScalarGlobalTopologyDofCount( fe_space ) );
      }
   }
   else if constexpr ( std::is_same_v< Restriction, H1Restriction > )
   {
      return static_cast< Integer >(
         ScalarGlobalTopologyDofCount( fe_space ) );
   }
   else if constexpr ( is_vector_h1_restriction_v< Restriction > )
   {
      static_assert(
         is_vector_shape_functions_v< ShapeFunctions >,
         "VectorH1Restriction requires a vector finite element space." );
      static_assert(
         Restriction::num_comp == ShapeFunctions::vector_dim,
         "VectorH1Restriction<NComp> must match the vector finite element component count." );

      return static_cast< Integer >( Restriction::num_comp ) *
         static_cast< Integer >(
            ScalarGlobalTopologyDofCount( fe_space ) );
   }
   else if constexpr ( is_tensor_product_restriction_v< Restriction > )
   {
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "TensorProductRestriction v1 supports scalar finite element spaces only." );
      return static_cast< Integer >(
         ScalarGlobalTopologyDofCount( fe_space ) );
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "FiniteElementDofCount supports only L2Restriction, scalar H1Restriction, VectorH1Restriction, and scalar TensorProductRestriction." );
      return 0;
   }
}

template < typename FESpace >
GENDIL_HOST_DEVICE
GlobalIndex ZeroBasedElementToGlobalDofIndex(
   const FESpace & fe_space,
   const GlobalIndex element_index,
   const GlobalIndex scalar_local_dof_index )
{
   using Space = std::remove_cvref_t< FESpace >;
   using Restriction = typename Space::restriction_type;
   using ShapeFunctions =
      typename Space::finite_element_type::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "ZeroBasedElementToGlobalDofIndex supports scalar finite element spaces only." );

   if constexpr ( std::is_same_v< Restriction, L2Restriction > )
   {
      constexpr GlobalIndex scalar_local_dofs =
         LocalDofCount< ShapeFunctions >();
      return element_index * scalar_local_dofs + scalar_local_dof_index;
   }
   else if constexpr ( std::is_same_v< Restriction, H1Restriction > )
   {
      constexpr GlobalIndex scalar_local_dofs =
         LocalDofCount< ShapeFunctions >();
      const GlobalIndex restriction_index =
         element_index * scalar_local_dofs + scalar_local_dof_index;
      const int global_index = fe_space.restriction.indices[restriction_index];
      GENDIL_VERIFY(
         global_index >= 0,
         "H1Restriction contains a negative element-to-global DoF index." );
      return static_cast< GlobalIndex >( global_index );
   }
   else if constexpr ( is_tensor_product_restriction_v< Restriction > )
   {
      return TensorProductElementToGlobalDofIndex(
         fe_space.restriction,
         element_index,
         scalar_local_dof_index );
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "ZeroBasedElementToGlobalDofIndex supports only scalar L2Restriction, H1Restriction, and TensorProductRestriction." );
      return 0;
   }
}

template < typename FESpace >
GENDIL_HOST_DEVICE
GlobalIndex ScalarElementDofOrdinalToGlobalDofIndex(
   const FESpace & fe_space,
   const GlobalIndex flat_element_dof_ordinal )
{
   using Space = std::remove_cvref_t< FESpace >;
   using Restriction = typename Space::restriction_type;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "ScalarElementDofOrdinalToGlobalDofIndex supports scalar finite element spaces only." );

   if constexpr ( std::is_same_v< Restriction, L2Restriction > )
   {
      return fe_space.restriction.shift + flat_element_dof_ordinal;
   }
   else if constexpr ( std::is_same_v< Restriction, H1Restriction > )
   {
      const int global_index =
         fe_space.restriction.indices[flat_element_dof_ordinal];
      GENDIL_VERIFY(
         global_index >= 0,
         "H1Restriction contains a negative element-to-global DoF index." );
      return static_cast< GlobalIndex >( global_index );
   }
   else if constexpr ( is_tensor_product_restriction_v< Restriction > )
   {
      constexpr GlobalIndex scalar_local_dofs =
         LocalDofCount< ShapeFunctions >();
      const GlobalIndex element_index =
         flat_element_dof_ordinal / scalar_local_dofs;
      const GlobalIndex scalar_local_dof_index =
         flat_element_dof_ordinal - element_index * scalar_local_dofs;
      return TensorProductElementToGlobalDofIndex(
         fe_space.restriction,
         element_index,
         scalar_local_dof_index );
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "ScalarElementDofOrdinalToGlobalDofIndex supports only scalar L2Restriction, H1Restriction, and TensorProductRestriction." );
      return 0;
   }
}

template < typename FESpace >
GENDIL_HOST_DEVICE
GlobalIndex GlobalDofIndex(
   const FESpace & fe_space,
   const GlobalIndex element_index,
   const GlobalIndex scalar_local_dof_index )
{
   using Space = std::remove_cvref_t< FESpace >;
   using Restriction = typename Space::restriction_type;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;
   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "Vector global DoF indexing requires a compile-time component tag." );

   if constexpr ( std::is_same_v< Restriction, L2Restriction > )
   {
      constexpr GlobalIndex scalar_local_dofs =
         LocalDofCount< ShapeFunctions >();
      const GlobalIndex flat_element_dof_ordinal =
         element_index * scalar_local_dofs + scalar_local_dof_index;
      return fe_space.restriction.shift + flat_element_dof_ordinal;
   }
   else if constexpr ( std::is_same_v< Restriction, H1Restriction > )
   {
      constexpr GlobalIndex scalar_local_dofs =
         LocalDofCount< ShapeFunctions >();
      const GlobalIndex flat_element_dof_ordinal =
         element_index * scalar_local_dofs + scalar_local_dof_index;
      const int global_index =
         fe_space.restriction.indices[flat_element_dof_ordinal];
      GENDIL_VERIFY(
         global_index >= 0,
         "H1Restriction contains a negative element-to-global DoF index." );
      return static_cast< GlobalIndex >( global_index );
   }
   else if constexpr ( is_tensor_product_restriction_v< Restriction > )
   {
      return TensorProductElementToGlobalDofIndex(
         fe_space.restriction,
         element_index,
         scalar_local_dof_index );
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "GlobalDofIndex supports only scalar L2Restriction, H1Restriction, and TensorProductRestriction." );
      return 0;
   }
}

template <
   typename FESpace,
   size_t Component >
GENDIL_HOST_DEVICE
GlobalIndex GlobalDofIndex(
   const FESpace & fe_space,
   std::integral_constant< size_t, Component > component,
   const GlobalIndex element_index,
   const GlobalIndex component_local_dof_index )
{
   using Space = std::remove_cvref_t< FESpace >;
   using Restriction = typename Space::restriction_type;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;

   if constexpr ( !is_vector_shape_functions_v< ShapeFunctions > )
   {
      static_assert(Component == 0, "Scalar finite element spaces only have component 0.");
      return GlobalDofIndex(
         fe_space,
         element_index,
         component_local_dof_index );
   }
   else if constexpr ( std::is_same_v< Restriction, L2Restriction > )
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
         component_local_dof_index;
   }
   else if constexpr ( is_vector_h1_restriction_v< Restriction > )
   {
      static_assert(
         is_vector_shape_functions_v< ShapeFunctions >,
         "VectorH1Restriction requires a vector finite element space." );
      static_assert(
         Restriction::num_comp == ShapeFunctions::vector_dim,
         "VectorH1Restriction<NComp> must match the vector finite element component count." );
      static_assert(
         Component < Restriction::num_comp,
         "VectorH1Restriction component index is out of bounds." );
      static_assert(
         VectorComponentDofShapesMatchFirst< ShapeFunctions >(),
         "VectorH1Restriction currently requires identical scalar component DoF shapes." );

      using ComponentDofShape =
         component_dof_shape_t< ShapeFunctions, Component >;
      constexpr GlobalIndex component_local_dofs =
         Product( ComponentDofShape{} );
      const GlobalIndex restriction_index =
         element_index * component_local_dofs + component_local_dof_index;
      const int scalar_global_index =
         fe_space.restriction.indices[restriction_index];
      GENDIL_VERIFY(
         scalar_global_index >= 0,
         "VectorH1Restriction contains a negative scalar element-to-global DoF index." );

      return static_cast< GlobalIndex >( Component ) *
            static_cast< GlobalIndex >( fe_space.restriction.scalar_num_dofs ) +
         static_cast< GlobalIndex >( scalar_global_index );
   }
   else
   {
      static_assert(
         dependent_false_v< Restriction >,
         "GlobalDofIndex supports only L2Restriction, scalar H1Restriction, and VectorH1Restriction." );
      return 0;
   }
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
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   const GlobalIndex component_local_id =
      FlattenComponentLocalDof< ShapeFunctions >( component, indices );
   return GlobalDofIndex(
      fe_space,
      component,
      element_index,
      component_local_id );
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
   const GlobalIndex scalar_local_id =
      FlattenLocalDof< ShapeFunctions >( indices );
   return GlobalDofIndex(
      fe_space,
      element_index,
      scalar_local_id );
}

} // namespace gendil
