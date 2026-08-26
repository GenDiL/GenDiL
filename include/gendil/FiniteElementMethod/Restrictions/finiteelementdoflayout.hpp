// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Static finite-element DoF shapes, counts, offsets, and flattening.
 *
 * These utilities describe element-local layout independently of any
 * element-to-global restriction representation. Scalar local coordinates use
 * the repository's first-coordinate-fastest @c std::array convention; vector
 * layouts additionally use a compile-time component tag.
 */

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/finiteelementorders.hpp"
#include "gendil/Meshes/Geometries/point.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/vectorshapefunctions.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/cat.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/get.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/Utilities/multiindex.hpp"

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Element-local tensor coordinate tagged by a compile-time component.
 *
 * The component has no runtime storage and may represent either a physical
 * vector component or another statically selected DoF family.
 */
template < size_t Component, Integer TensorDim >
struct LocalComponentDoFIndex
{
   static constexpr size_t component = Component;

   std::array< GlobalIndex, TensorDim > local_dof;
};

namespace details {

template < typename Shape, Integer Dim, size_t... I >
GENDIL_HOST_DEVICE
constexpr bool DofIndexIsInBounds_impl(
   const std::array< GlobalIndex, Dim > & index,
   std::index_sequence< I... > )
{
   return
      ( ( index[I] < static_cast< GlobalIndex >( seq_get_v< I, Shape > ) ) &&
        ... );
}

/** @brief Return whether a local tensor coordinate lies within a static DoF shape. */
template < typename Shape, Integer Dim >
GENDIL_HOST_DEVICE
constexpr bool DofIndexIsInBounds(
   const std::array< GlobalIndex, Dim > & index )
{
   static_assert(
      Dim == static_cast< Integer >( Shape::size() ),
      "Local DoF index rank must match the static DoF shape." );
   return DofIndexIsInBounds_impl< Shape >(
      index,
      std::make_index_sequence< Dim >{} );
}

} // namespace details

/** @brief Derive the tensor DoF shape associated with scalar shape functions. */
template < typename ShapeFunctions >
struct FiniteElementDofShape
{
   using type = orders_to_num_dofs< typename ShapeFunctions::orders >;
};

/** @brief Preserve the tuple of component DoF shapes for vector shape functions. */
template < typename ... ScalarShapeFunctions >
struct FiniteElementDofShape< VectorShapeFunctions< ScalarShapeFunctions... > >
{
   using type = typename VectorShapeFunctions< ScalarShapeFunctions... >::dof_shape;
};

/** @brief DoF shape metadata associated with a shape-function type. */
template < typename ShapeFunctions >
using finite_element_dof_shape_t =
   typename FiniteElementDofShape< ShapeFunctions >::type;

/** @brief Select one component DoF shape from scalar or vector shape functions. */
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

/** @brief DoF tensor shape of compile-time component @c Component. */
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

/**
 * @brief Return whether every vector component has the first component's DoF
 * shape.
 *
 * Scalar shape functions return @c false.
 */
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

/** @brief Sum the DoF counts of selected shapes in a shape tuple. */
template < typename DofShapes, size_t ... I >
GENDIL_HOST_DEVICE
constexpr GlobalIndex DofShapeTupleDofCount( std::index_sequence< I... > )
{
   return ( GlobalIndex{0} + ... +
      Product( std::tuple_element_t< I, DofShapes >{} ) );
}

/** @brief Return the number of DoFs in one element, summed over components. */
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

/** @brief Return the compile-time local DoF count of a finite-element space. */
template < typename FESpace >
GENDIL_HOST_DEVICE
constexpr GlobalIndex LocalDofCount( const FESpace & )
{
   using ShapeFunctions =
      typename std::remove_cvref_t< FESpace >::finite_element_type::shape_functions;
   return LocalDofCount< ShapeFunctions >();
}

/**
 * @brief Return a component's prefix in component-major element-local
 * numbering.
 */
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

/** @brief Flatten coordinates within one statically selected component. */
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

/**
 * @brief Flatten a component-local coordinate into component-major
 * element-local numbering.
 */
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

/** @brief Flatten a scalar element-local tensor coordinate. */
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

/**
 * @brief Flatten a component-local coordinate using a finite-element space's
 * shape functions.
 */
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

/** @brief Flatten a scalar local coordinate using a finite-element space. */
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

template < typename ShapeFunctions, size_t Component, size_t... I >
GlobalIndex CheckedVectorComponentOffset_impl(
   const GlobalIndex num_elements,
   std::index_sequence< I... > )
{
   GlobalIndex offset = 0;
   ( ( offset = CheckedAdd(
          offset,
          CheckedMultiply(
             num_elements,
             static_cast< GlobalIndex >(
                Product(
                   component_dof_shape_t< ShapeFunctions, I >{} ) ),
             "Vector restriction component size overflow." ),
          "Vector restriction component prefix overflow." ) ),
     ... );
   return offset;
}

/**
 * @brief Return a checked component prefix over all element-local
 * occurrences.
 */
template < typename ShapeFunctions, size_t Component >
GlobalIndex CheckedVectorComponentOffset(
   const GlobalIndex num_elements )
{
   return CheckedVectorComponentOffset_impl< ShapeFunctions, Component >(
      num_elements,
      std::make_index_sequence< Component >{} );
}

/**
 * @brief Return the checked total number of element-local DoF occurrences.
 *
 * For vector shape functions this is the sum of the independently shaped
 * component counts over all elements.
 */
template < typename ShapeFunctions >
GlobalIndex CheckedElementLocalDofCount(
   const GlobalIndex num_elements )
{
   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      GlobalIndex total = 0;
      [&]< size_t... Component >( std::index_sequence< Component... > )
      {
         ( ( total = CheckedAdd(
                total,
                CheckedMultiply(
                   num_elements,
                   static_cast< GlobalIndex >(
                      Product(
                         component_dof_shape_t<
                            ShapeFunctions,
                            Component >{} ) ),
                   "Vector restriction local DoF count overflow." ),
                "Vector restriction local DoF sum overflow." ) ),
           ... );
      }( std::make_index_sequence< ShapeFunctions::vector_dim >{} );
      return total;
   }
   else
   {
      return CheckedMultiply(
         num_elements,
         static_cast< GlobalIndex >(
            Product( finite_element_dof_shape_t< ShapeFunctions >{} ) ),
         "Scalar restriction local DoF count overflow." );
   }
}

} // namespace gendil
