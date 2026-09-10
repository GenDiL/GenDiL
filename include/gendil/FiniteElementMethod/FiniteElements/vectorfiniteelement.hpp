// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Component-vector finite-element construction and selection.
 */

#include "gendil/FiniteElementMethod/FiniteElements/finiteelement.hpp"
#include "gendil/Meshes/Geometries/point.hpp"
#include "gendil/Utilities/MathHelperFunctions/sum.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/vectorshapefunctions.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Combine scalar finite elements into one component-vector element.
 *
 * Components may use different scalar shape functions but must use the same
 * reference geometry.
 */
template<typename FirstFiniteElementType, typename... RestFiniteElementTypes>
auto MakeVectorFiniteElement(
   FirstFiniteElementType,
   RestFiniteElementTypes...)
{
   static_assert(
      (std::is_same_v<
         typename RestFiniteElementTypes::geometry,
         typename FirstFiniteElementType::geometry> && ...),
      "MakeVectorFiniteElement: component finite elements must use the same "
      "reference geometry type." );
   using geometry = typename FirstFiniteElementType::geometry;
   using shape_functions = VectorShapeFunctions<
      typename FirstFiniteElementType::shape_functions,
      typename RestFiniteElementTypes::shape_functions...>;
   return FiniteElement<geometry, shape_functions>{};
}

namespace finite_element_detail {

template<typename FiniteElementType>
using finite_element_shape_functions_t =
   typename std::remove_cvref_t<FiniteElementType>::shape_functions;

template<typename FiniteElementType>
inline constexpr bool is_vector_finite_element_v =
   is_vector_shape_functions_v<
      finite_element_shape_functions_t<FiniteElementType>>;

template<typename FiniteElementType>
consteval size_t VectorComponentCount()
{
   using ShapeFunctions =
      finite_element_shape_functions_t<FiniteElementType>;
   if constexpr (is_vector_shape_functions_v<ShapeFunctions>)
   {
      return static_cast<size_t>(ShapeFunctions::vector_dim);
   }
   else
   {
      return 0;
   }
}

template<size_t Component, typename FiniteElementType>
struct ComponentShapeFunctions
{
   using ShapeFunctions =
      finite_element_shape_functions_t<FiniteElementType>;
   static_assert(
      is_vector_shape_functions_v<ShapeFunctions>,
      "Finite-element component selection requires vector shape functions." );
   static_assert(
      Component < static_cast<size_t>(ShapeFunctions::vector_dim),
      "Finite-element component index is out of range." );
   using type = std::tuple_element_t<
      Component,
      typename ShapeFunctions::scalar_shape_functions_tuple>;
};

template<size_t Component, typename FiniteElementType>
using component_shape_functions_t =
   typename ComponentShapeFunctions<Component, FiniteElementType>::type;

template<typename FiniteElementType, size_t... Component>
auto MakeHomogeneousVectorFiniteElementImpl(
   const FiniteElementType& finite_element,
   std::index_sequence<Component...>)
{
   return MakeVectorFiniteElement(
      (static_cast<void>(Component), finite_element)...);
}

} // namespace finite_element_detail

/**
 * @brief Repeat one scalar finite element into a component-vector element.
 *
 * Unlike `MakeVectorFiniteElement`, every component uses the same scalar
 * finite-element type.
 */
template<size_t NumComponents, typename FiniteElementType>
auto MakeHomogeneousVectorFiniteElement(
   const FiniteElementType& finite_element)
{
   static_assert(
      NumComponents > 0,
      "MakeHomogeneousVectorFiniteElement requires at least one component." );
   static_assert(
      !is_vector_shape_functions_v<
         typename FiniteElementType::shape_functions>,
      "MakeHomogeneousVectorFiniteElement requires a scalar finite element." );
   return finite_element_detail::MakeHomogeneousVectorFiniteElementImpl(
      finite_element,
      std::make_index_sequence<NumComponents>{});
}

/** @brief Select one scalar component from a component-vector finite element. */
template<size_t Component, typename FiniteElementType>
auto GetComponentFiniteElement(const FiniteElementType&)
{
   using VectorFiniteElement = std::remove_cvref_t<FiniteElementType>;
   using ShapeFunctions =
      finite_element_detail::component_shape_functions_t<
         Component,
         VectorFiniteElement>;
   return FiniteElement<
      typename VectorFiniteElement::geometry,
      ShapeFunctions>{};
}

} // namespace gendil
