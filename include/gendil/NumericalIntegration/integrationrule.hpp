// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MathHelperFunctions/sum.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "QuadraturePoints/numpoints.hpp"
#include "QuadraturePoints/tensorpoints.hpp"
#include "QuadraturePoints/GLpoints.hpp"
#include "QuadraturePoints/GLLpoints.hpp"

namespace gendil {

/**
 * @brief A structure representing a quadrature rule for integrating functions.
 * 
 * @tparam Geometry The reference geometry describing the integral domain.
 * @tparam QuadraturePointSet The set of quadrature weights and points.
 */
template < typename Geometry, typename QuadraturePointSet >
struct IntegrationRule
{
   using geometry = Geometry;
   using points = QuadraturePointSet; // This is a tuple for tensor elements
   using index_type = typename QuadraturePointSet::index_type;

   static constexpr Integer space_dim = Geometry::space_dim;
   static constexpr Integer geometry_dim = Geometry::geometry_dim;
   static constexpr Integer num_points = QuadraturePointSet::GetNumPoints();
   // FIXME: Add accuracy?

   /**
    * @brief Get the i-th quadrature point object.
    * 
    * @param i The index of the quadrature point.
    * @return The quadrature point.
    */
   GENDIL_HOST_DEVICE
   static constexpr Point< space_dim > GetPoint( index_type index )
   {
      return QuadraturePointSet::GetPoint( index );
   }

   // TODO: Make this a free-function?
   template < typename Lambda, size_t... Is >
   GENDIL_HOST_DEVICE
   static constexpr Real Integrate( Lambda && lambda, std::index_sequence<Is...> )
   {
      // return Sum( GetWeight<Is, points>() * lambda( GetPoint<Is, points>()... ) );
      return Sum( QuadraturePointSet::weight[Is] * lambda( QuadraturePointSet::points[Is] )... );
   }

   /**
    * @brief Integrate an arbitrary function using the integration rule.
    * The function must take a point as parameter.
    * 
    * @tparam Lambda The type of the function to integrate.
    * @param lambda The function to integrate. Must take a point as parameter.
    * @return The integral of the function.
    */
   template < typename Lambda >
   GENDIL_HOST_DEVICE
   static constexpr Real Integrate( Lambda && lambda )
   {
      return Integrate( lambda, std::make_index_sequence< num_points >{} );
   }
};

/**
 * @brief Factory to build tensor integration rules. Useful to hide explicit type.
 * 
 * @tparam NumPoints List of the number of integration points for each dimension.
 * @param num_points An empty variable representing a static list of number of integration points for each dimension.
 * @return auto The tensor integration rule.
 */
template < int... NumPoints >
auto MakeIntegrationRule( const IntegrationRuleNumPoints< NumPoints... > & num_points )
{
   constexpr Integer dim = sizeof...(NumPoints);

   using geometry = HyperCube< dim >;
   using points = TensorPoints< GaussLegendrePoints< NumPoints >... >;

   return IntegrationRule< geometry, points >{};
}

template < int... FirstNumPoints, int... NextNumPoints, typename... RestArgs >
auto MakeIntegrationRule( const IntegrationRuleNumPoints< FirstNumPoints... > & first_num_points,
                          const IntegrationRuleNumPoints< NextNumPoints... > & next_num_points,
                          RestArgs... rest_args )
{
   return MakeIntegrationRule( IntegrationRuleNumPoints< FirstNumPoints..., NextNumPoints... >{}, rest_args... );
}

template < int... NumPoints >
auto MakeLobattoIntegrationRule( const IntegrationRuleNumPoints< NumPoints... > & num_points )
{
   constexpr Integer dim = sizeof...(NumPoints);

   using geometry = HyperCube< dim >;
   using points = TensorPoints< GaussLobattoLegendrePoints< NumPoints >... >;

   return IntegrationRule< geometry, points >{};
}

template < int... FirstNumPoints, int... NextNumPoints, typename... RestArgs >
auto MakeLobattoIntegrationRule( const IntegrationRuleNumPoints< FirstNumPoints... > & first_num_points,
                                 const IntegrationRuleNumPoints< NextNumPoints... > & next_num_points,
                                 RestArgs... rest_args )
{
   return MakeLobattoIntegrationRule( IntegrationRuleNumPoints< FirstNumPoints..., NextNumPoints... >{}, rest_args... );
}

template < int Offset, int Size, typename Geometry, typename QuadraturePoints, size_t... Is >
auto GetSubIntegrationRule(
   IntegrationRule< Geometry, QuadraturePoints > const & int_rule,
   std::index_sequence<Is...> )
{
   // Assumes tensor rule
   using points = TensorPoints< std::tuple_element_t< Offset + Is, typename QuadraturePoints::points_1d_tuple >... >;
   return IntegrationRule< HyperCube<Size>, points >{};
}

template < int Offset, int Size, typename Geometry, typename QuadraturePointSet >
auto GetSubIntegrationRule( IntegrationRule< Geometry, QuadraturePointSet > const & int_rule )
{
   return GetSubIntegrationRule< Offset, Size >( int_rule, std::make_index_sequence< Size >{} );
}

template < Integer Offset, Integer Size, typename IntegrationRule >
struct SubIntegrationRule_t
{
   using type = decltype( GetSubIntegrationRule< Offset, Size >( IntegrationRule{} ) );
};

template < Integer Offset, Integer Size, typename IntegrationRule >
using SubIntegrationRule = typename SubIntegrationRule_t< Offset, Size, IntegrationRule >::type;

}
