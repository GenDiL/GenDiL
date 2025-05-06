// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/getorder.hpp"
#include "tensorpoints.hpp"

namespace gendil {

template < int... NumPoints >
struct IntegrationRuleNumPoints
{
   static constexpr Integer Dim = sizeof...( NumPoints );
};

template < typename Points >
struct get_num_points;
// {
//    static constexpr Integer value = Points::num_points;
// };

template < typename... Points >
struct get_num_points< std::tuple< Points... > >
{
   using type = IntegrationRuleNumPoints< Points::num_points... >;
};

template < typename... Points >
struct get_num_points< TensorPoints< Points... > >
{
   using type = typename get_num_points< typename TensorPoints< Points... >::points >::type;
};

template < typename Points >
using get_num_points_t = typename get_num_points< Points >::type;

template < typename NumPoints, Integer index>
struct get_num_points_of_dim_v;

template < Integer index, Integer... NumPoints > 
struct get_num_points_of_dim_v< IntegrationRuleNumPoints< NumPoints... >, index >
{
   static constexpr Integer value = get_order< index, NumPoints... >::value;
};

template < Integer index, Integer... NumPoints > 
struct get_num_points_of_dim_v< std::integer_sequence< Integer, NumPoints... >, index >
{
   static constexpr Integer value = get_order< index, NumPoints... >::value;
};

// template < typename NumPoints, int index>
// constexpr int get_num_points_of_dim = get_num_points_of_dim_v< NumPoints, index >::value;

}
