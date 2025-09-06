// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MathHelperFunctions/sum.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tuplereplace.hpp"
#include "gendil/Meshes/Geometries/geometries.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "QuadraturePoints/zeropoint.hpp"
#include "QuadraturePoints/onepoint.hpp"

namespace gendil {

/**
 * @brief Get the integretion rules on the faces of an hyper-cube based on the provided "volume" integration rule.
 * 
 * @tparam IntegrationRule The type of the input integration rule.
 * @param int_rule The "volume" integration rule.
 * @return A tuple of face integration rules.
 */
template < typename IntegrationRule >
constexpr auto GetFaceIntegrationRules( IntegrationRule const & int_rule );

// TODO: Add doc
template < size_t index, typename CurrentIntegrationRule, typename Points >
constexpr auto replace( CurrentIntegrationRule const & int_rule, Points const & new_points_1d )
{
   // TODO: explain
   using new_points = decltype( tuple_replace< index >( typename CurrentIntegrationRule::points::points_1d_tuple{}, new_points_1d ) );
   using point_set = decltype( MakeTensorPoints( new_points{} ) );
   constexpr Integer geometry_dim = CurrentIntegrationRule::geometry::geometry_dim;
   constexpr Integer space_dim = CurrentIntegrationRule::geometry::space_dim;
   return IntegrationRule< HyperCube< geometry_dim - 1, space_dim >, point_set >{};
}

// TODO: Rename, this is very confusing
template < typename NewPoints, typename IntegrationRule, size_t... Is >
constexpr auto GetFaceIntegrationRules( IntegrationRule const & int_rule, std::index_sequence<Is...> )
{
   return std::make_tuple( replace<Is>( int_rule, NewPoints{} )... );
}

// TODO: Add enable_if_t Hypercube
// TODO: Add a note on the ordering of the face integration rule for Hypercube.
// TODO: Add impl namespace
template < typename IntegrationRule >
constexpr auto GetFaceIntegrationRules( IntegrationRule const & int_rule )
{
   // The following has implication on the order of the faces.
   return   std::tuple_cat(
               GetFaceIntegrationRules< ZeroPoint >(
                  int_rule,
                  std::make_index_sequence< IntegrationRule::space_dim >{}
               ),
               GetFaceIntegrationRules< OnePoint >(
                  int_rule,
                  std::make_index_sequence< IntegrationRule::space_dim >{}
               )
            );
}

/**
 * @brief Get the low dimension face integration rules.
 * 
 * @tparam LowDim 
 * @tparam FaceIntegrationRule 
 * @param face_int_rules 
 * @return constexpr auto 
 */
template < Integer LowDim, typename FaceIntegrationRule >
constexpr auto GetLowDimFaceIntegrationRules( FaceIntegrationRule const & face_int_rules );

template < Integer LowDim, typename HighDimIntegrationRule, size_t... Is >
constexpr auto GetLowDimIntegrationRule( HighDimIntegrationRule const & int_rule, std::index_sequence<Is...> )
{
   using high_dim_points = typename HighDimIntegrationRule::points::points_1d_tuple;
   using low_dim_points = std::tuple< std::tuple_element_t< Is, high_dim_points >... >;
   using point_set = decltype( MakeTensorPoints( low_dim_points{} ) );
   constexpr Integer geometry_dim = HighDimIntegrationRule::geometry::geometry_dim;
   return IntegrationRule< HyperCube< geometry_dim, LowDim >, point_set >{};
}

template < Integer LowDim, typename HighDimIntegrationRule >
constexpr auto GetLowDimIntegrationRule( HighDimIntegrationRule const & int_rule )
{
   return GetLowDimIntegrationRule< LowDim >( int_rule, std::make_index_sequence< LowDim >{} );
}

template < Integer LowDim, typename FaceIntegrationRule, size_t... Is >
constexpr auto GetLowDimFaceIntegrationRules( FaceIntegrationRule const & face_int_rules, std::index_sequence<Is...> )
{
   return std::make_tuple( GetLowDimIntegrationRule< LowDim >( std::get< Is >( face_int_rules ) )... );
}

template < Integer LowDim, typename FaceIntegrationRule >
constexpr auto GetLowDimFaceIntegrationRules( FaceIntegrationRule const & face_int_rules )
{
   constexpr Integer size = std::tuple_size_v< FaceIntegrationRule >;
   return GetLowDimFaceIntegrationRules< LowDim >( face_int_rules, std::make_index_sequence< size >{} );
}

template <
   CellFaceView FaceInfo,
   typename FaceIntegrationRule
>
constexpr auto GetFaceIntegrationRule( const FaceInfo & face_info, FaceIntegrationRule const & face_int_rules )
{
   constexpr Integer local_face_index = FaceInfo::local_face_index_type::value;
   return std::get< local_face_index >( face_int_rules );
}

}
