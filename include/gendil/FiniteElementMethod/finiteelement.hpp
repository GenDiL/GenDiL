// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/getorder.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "ShapeFunctions/GLLshapefunctions.hpp"
#include "ShapeFunctions/GLshapefunctions.hpp"
#include "ShapeFunctions/tensorshapefunctions.hpp"
#include "gendil/Meshes/Geometries/geometries.hpp"

namespace gendil {

// TODO: Replace Geometry by ReferenceCell?
/**
 * @brief A simple struct representing a finite element.
 * 
 * @tparam Geometry The geometry of the finite element.
 * @tparam ShapeFunctions The shape functions associated to the finite element.
 */
template < typename Geometry, typename ShapeFunctions >
struct FiniteElement
{
   using geometry = Geometry;
   using shape_functions = ShapeFunctions;
   static constexpr Integer space_dim = Geometry::space_dim;
   static constexpr Integer geometry_dim = Geometry::geometry_dim;

   constexpr FiniteElement()
   {
      static_assert(
         geometry_dim == shape_functions::dim,
         "Element topology dimension and element shape functions dimension mismatch."
      );
   }

   static constexpr Integer GetNumDofs()
   {
      return shape_functions::num_dofs;
   }
};

/**
 * @brief A factory to construct finite elements with tensor shape functions and anisotropic polynomial orders.
 * 
 * @tparam ShapeFunctions The shape functions collection used in each dimension.
 * @tparam Orders The list of polynomial orders used for each dimension.
 * @return auto The resulting finite element.
 */
template < template <Integer Order> typename ShapeFunctions,
           Integer... Orders >
auto MakeFiniteElement( FiniteElementOrders< Orders... > )
{
   using topology = HyperCube< sizeof...( Orders ) >;
   using shape_functions = TensorShapeFunctions< ShapeFunctions< Orders >... >;
   return FiniteElement< topology, shape_functions >();
}

template < template <Integer Order> typename ShapeFunctions,
           Integer... FirstOrders,
           Integer... NextOrders,
           typename... Args >
auto MakeFiniteElement( FiniteElementOrders< FirstOrders... >, FiniteElementOrders< NextOrders... >, Args... rest_orders )
{
   return MakeFiniteElement<ShapeFunctions>( FiniteElementOrders< FirstOrders..., NextOrders... >{}, rest_orders... );
}

/**
 * @brief Type alias for Gauss-Lobatto-Legendre (GLL) finite elements with anisotropic polynomial orders.
 * 
 * @tparam Orders The list of polynomial orders used for each dimension.
 */
template < Integer... Orders >
using GLLFiniteElement =   FiniteElement<
                              HyperCube< sizeof...(Orders) >,
                              TensorShapeFunctions< GaussLobattoLegendreShapeFunctions< Orders >... >
                           >;

/**
 * @brief Type alias for Gauss-Legendre (GL) finite elements with anisotropic polynomial orders.
 * 
 * @tparam Orders The list of polynomial orders used for each dimension.
 */
template < Integer... Orders >
using GLFiniteElement = FiniteElement<
                           HyperCube< sizeof...(Orders) >,
                           TensorShapeFunctions< GaussLegendreShapeFunctions< Orders >... >
                        >;

/**
 * @brief Create a finite element using the provided a list of approximation orders.
 * 
 * @tparam Orders The list of approximation orders provided as a list of FiniteElementOrders.
 * @param orders Empty variable used to provide the approximation orders.
 * @return auto 
 */
template < typename... Orders >
auto MakeLegendreFiniteElement( Orders... orders )
{
   return MakeFiniteElement<GaussLegendreShapeFunctions>( orders... );
}

template < typename... Orders >
auto MakeLobattoFiniteElement( Orders... orders )
{
   return MakeFiniteElement<GaussLobattoLegendreShapeFunctions>( orders... );
}

}
