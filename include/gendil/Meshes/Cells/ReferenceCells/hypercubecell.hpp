// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Meshes/Geometries/point.hpp"

namespace gendil {

/**
 * @brief A simple structure representing a stretched hypercube cell.
 * 
 */
template < Integer Dim_ >
struct HyperCubeCell
{
    static constexpr Integer Dim = Dim_;

    const Point<Dim> origin;
    const std::array< Real, Dim > h;

   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::array< Real, Dim >;
   template < typename IntRule >
   using QuadData = typename IntRule::points::points_1d_tuple;

   GENDIL_HOST_DEVICE
   HyperCubeCell(
      Point< Dim > origin,
      std::array< Real, Dim > h ) :
         origin( origin ),
         h( h )
   { }

   template < typename QuadData >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( TensorIndex< Dim > const & quad_index,
                              QuadData const & quad_data,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      ConstexprLoop< Dim >( [&] ( auto i )
      {
         using quad = std::tuple_element_t< i, QuadData >;
         const auto q_pt = quad::GetCoord( quad_index[i] );
         X[i] = origin[i] + h[i] * q_pt;
         J_mesh[i] = h[i];
      });
   }

   static constexpr Integer GetNormalDimensionIndex( Integer face_index )
   {
      return face_index % Dim;
   }

   static constexpr int GetNormalSign( Integer face_index )
   {
      return face_index < Dim ? -1 : 1;
   }
};

}
