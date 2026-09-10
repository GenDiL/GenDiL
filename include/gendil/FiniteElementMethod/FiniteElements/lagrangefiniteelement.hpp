// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Tensor-product Legendre and Lobatto finite-element construction.
 */

#include "gendil/FiniteElementMethod/FiniteElements/finiteelement.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/finiteelementorders.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/GLLshapefunctions.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/GLshapefunctions.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/FiniteElementMethod/ShapeFunctions/tensorshapefunctions.hpp"

namespace gendil {

/**
 * @brief Construct an anisotropic hypercube finite element.
 *
 * @tparam ShapeFunctions One-dimensional shape-function family instantiated
 *                        independently for every axis.
 * @tparam Orders Polynomial orders along the element axes.
 */
template<template<Integer Order> typename ShapeFunctions, Integer... Orders>
auto MakeFiniteElement(FiniteElementOrders<Orders...>)
{
   using topology = HyperCube<sizeof...(Orders)>;
   using shape_functions =
      TensorShapeFunctions<ShapeFunctions<Orders>...>;
   return FiniteElement<topology, shape_functions>();
}

/** Join several order packs before constructing the finite element. */
template<template<Integer Order> typename ShapeFunctions,
         Integer... FirstOrders,
         Integer... NextOrders,
         typename... Args>
auto MakeFiniteElement(
   FiniteElementOrders<FirstOrders...>,
   FiniteElementOrders<NextOrders...>,
   Args... rest_orders)
{
   return MakeFiniteElement<ShapeFunctions>(
      FiniteElementOrders<FirstOrders..., NextOrders...>{},
      rest_orders...);
}

/**
 * @brief Gauss-Lobatto-Legendre finite element with anisotropic orders.
 *
 * Lobatto nodes include element endpoints and are commonly used for nodal H1
 * spaces when paired with a conforming restriction.
 */
template<Integer... Orders>
using GLLFiniteElement = FiniteElement<
   HyperCube<sizeof...(Orders)>,
   TensorShapeFunctions<GaussLobattoLegendreShapeFunctions<Orders>...>>;

/**
 * @brief Gauss-Legendre finite element with anisotropic orders.
 *
 * Legendre nodes are interior to each element and are commonly paired with a
 * discontinuous restriction.
 */
template<Integer... Orders>
using GLFiniteElement = FiniteElement<
   HyperCube<sizeof...(Orders)>,
   TensorShapeFunctions<GaussLegendreShapeFunctions<Orders>...>>;

/** Construct a tensor-product Gauss-Legendre finite element. */
template<typename... Orders>
auto MakeLegendreFiniteElement(Orders... orders)
{
   return MakeFiniteElement<GaussLegendreShapeFunctions>(orders...);
}

/** Construct a tensor-product Gauss-Lobatto-Legendre finite element. */
template<typename... Orders>
auto MakeLobattoFiniteElement(Orders... orders)
{
   return MakeFiniteElement<GaussLobattoLegendreShapeFunctions>(orders...);
}

} // namespace gendil
