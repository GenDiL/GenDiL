// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Meshes/Geometries/point.hpp"

namespace gendil {

// !FIXME: I think this class doesn't really make sense and creates issues.
/**
 * @brief Structure representing a high-dimension quadrature point set constructed by
 * cartesian product of 1D quadrature points.
 * 
 * @tparam PointSets The different 1D point sets.
 */
template < typename... PointSets >
struct TensorPoints
{
   static constexpr Integer dim = sizeof...( PointSets );
   static constexpr Integer num_points = Product( PointSets::GetNumPoints()... );
   // ?Is it really only limited to 1d points?
   using points_1d_tuple = std::tuple< PointSets... >;
   using num_points_tensor = std::integer_sequence< Integer, PointSets::GetNumPoints()... >;
   using index_type = TensorIndex< dim >;

   template < Integer... Is >
   static constexpr Point< dim > GetPoint( const index_type & index, std::index_sequence< Is... > )
   {
      return Point< dim >{ std::tuple_element_t< Is, points_1d_tuple >::GetPoint( index[Is] )... };
   }

   static constexpr Point< dim > GetPoint( const index_type & index )
   {
      return GetPoint( index, std::make_index_sequence< dim >{} );
   }

   static constexpr Integer GetNumPoints()
   {
      return num_points;
   }
};

template < typename... Points >
auto MakeTensorPoints( std::tuple< Points ... > const & )
{
   return TensorPoints< Points... >{};
}

}
