// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Meshes/Geometries/point.hpp"

namespace gendil {

/**
 * @brief A simple structure representing a stretched segment cell.
 * 
 */
struct SegmentCell
{
   static constexpr Integer Dim = 1;

   const Real h; // element size
   const Point<1> cell_origin; // cell origin.

   GENDIL_HOST_DEVICE
   SegmentCell( Point<1> origin, Real h ) :
      h( h ), cell_origin( origin )
   {
      // TODO: Some checks on the mesh to verify that it's a line mesh?
   }

   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::array< std::array< Real, Dim >, Dim >;
   template < typename IntRule >
   using QuadData =  std::tuple<
                        std::tuple_element_t<0, typename IntRule::points::points_1d_tuple >
                     >;

   template < typename QuadData >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( const TensorIndex< Dim > & quad_index,
                              const QuadData & quad_data,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      GetValuesAndJacobian( quad_index[0], quad_data, X, J_mesh );
   }

   // can this be constexpr ?
   template < typename QuadData >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( const LocalIndex qx,
                              const QuadData & quad_data,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      using QuadType = std::tuple_element_t<0, QuadData>;
      X[0] = cell_origin + h * QuadType::GetCoord( qx );
      J_mesh[0][0] = h;
   }
};

}
