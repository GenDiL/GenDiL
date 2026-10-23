// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Tensor composition of scalar and component-vector finite elements.
 */

#include "gendil/FiniteElementMethod/FiniteElements/vectorfiniteelement.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/tensorshapefunctions.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

namespace finite_element_detail {

template<typename Tuple>
struct TensorFiniteElementFromAxes;

template<typename... AxisShapeFunctions>
struct TensorFiniteElementFromAxes<std::tuple<AxisShapeFunctions...>>
{
   using type = FiniteElement<
      HyperCube<static_cast<Integer>(sizeof...(AxisShapeFunctions))>,
      TensorShapeFunctions<AxisShapeFunctions...>>;
};

template<typename ShapeFunctions>
struct TensorAxes;

template<typename... AxisShapeFunctions>
struct TensorAxes<TensorShapeFunctions<AxisShapeFunctions...>>
{
   using type = std::tuple<AxisShapeFunctions...>;
};

template<typename FiniteElementType>
consteval bool IsSupportedTensorFactor()
{
   using FiniteElementTypeWithoutCV =
      std::remove_cvref_t<FiniteElementType>;
   using ShapeFunctions =
      typename FiniteElementTypeWithoutCV::shape_functions;
   constexpr bool geometry_is_hypercube =
      is_hypercube_geometry<
         typename FiniteElementTypeWithoutCV::geometry>::value;
   if constexpr (is_vector_shape_functions_v<ShapeFunctions>)
   {
      return geometry_is_hypercube && []<size_t... Component>(
         std::index_sequence<Component...>)
      {
         return (
            is_tensor_shape_functions_v<
               component_shape_functions_t<
                  Component,
                  FiniteElementType>> && ...);
      }(std::make_index_sequence<ShapeFunctions::vector_dim>{});
   }
   else
   {
      return geometry_is_hypercube &&
         is_tensor_shape_functions_v<ShapeFunctions>;
   }
}

template<size_t Component,
         typename FiniteElementType,
         bool IsVector = is_vector_finite_element_v<FiniteElementType>>
struct SelectedTensorShapeFunctions
{
   using type = finite_element_shape_functions_t<FiniteElementType>;
};

template<size_t Component, typename FiniteElementType>
struct SelectedTensorShapeFunctions<Component, FiniteElementType, true>
{
   using type = component_shape_functions_t<Component, FiniteElementType>;
};

template<size_t Component, typename FiniteElementType>
using selected_tensor_shape_functions_t =
   typename SelectedTensorShapeFunctions<
      Component,
      FiniteElementType>::type;

template<size_t Component, typename... FactorFiniteElements>
auto MakeScalarTensorFiniteElement(const FactorFiniteElements&...)
{
   using Axes = decltype(std::tuple_cat(
      std::declval<typename TensorAxes<
         selected_tensor_shape_functions_t<
            Component,
            FactorFiniteElements>>::type>()...));
   return typename TensorFiniteElementFromAxes<Axes>::type{};
}

template<size_t NumComponents,
         typename... FactorFiniteElements,
         size_t... Component>
auto MakeVectorTensorFiniteElement(
   std::index_sequence<Component...>,
   const FactorFiniteElements&... factors)
{
   static_assert(sizeof...(Component) == NumComponents);
   return MakeVectorFiniteElement(
      MakeScalarTensorFiniteElement<Component>(factors...)...);
}

template<typename... FactorFiniteElements>
consteval size_t TensorVectorComponentCount()
{
   size_t result = 0;
   ((result = result == 0
      ? VectorComponentCount<FactorFiniteElements>()
      : result), ...);
   return result;
}

template<size_t NumComponents, typename FiniteElementType>
consteval bool TensorVectorComponentCountMatches()
{
   constexpr size_t count = VectorComponentCount<FiniteElementType>();
   return count == 0 || count == NumComponents;
}

} // namespace finite_element_detail

/**
 * @brief Form the tensor product of current hypercube tensor finite elements.
 *
 * Scalar factors are broadcast across vector factors. Multiple vector factors
 * are zipped componentwise and must have the same component count. The
 * current implementation flattens the axes of existing tensor shape
 * functions; it does not define primitive multidimensional factor semantics.
 */
template<typename... FactorFiniteElements>
auto MakeTensorFiniteElement(const FactorFiniteElements&... factors)
{
   static_assert(
      sizeof...(FactorFiniteElements) > 0,
      "MakeTensorFiniteElement requires at least one factor." );
   static_assert(
      (finite_element_detail::IsSupportedTensorFactor<
         FactorFiniteElements>() && ...),
      "MakeTensorFiniteElement currently requires hypercube finite elements "
      "with scalar or component-vector TensorShapeFunctions." );

   constexpr size_t num_components =
      finite_element_detail::TensorVectorComponentCount<
         FactorFiniteElements...>();
   constexpr bool matching_components =
      (finite_element_detail::TensorVectorComponentCountMatches<
         num_components,
         FactorFiniteElements>() && ...);
   static_assert(
      matching_components,
      "Every vector MakeTensorFiniteElement factor must have the same "
      "component count." );

   if constexpr (num_components == 0)
   {
      return finite_element_detail::MakeScalarTensorFiniteElement<0>(
         factors...);
   }
   else
   {
      return finite_element_detail::MakeVectorTensorFiniteElement<
         num_components>(
            std::make_index_sequence<num_components>{},
            factors...);
   }
}

} // namespace gendil
