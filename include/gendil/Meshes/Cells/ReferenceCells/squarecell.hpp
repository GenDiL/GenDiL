// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Meshes/Geometries/point.hpp"
#include "gendil/NumericalIntegration/QuadraturePoints/getcoord.hpp"

namespace gendil {

/**
 * @brief A simple structure representing a stretched square cell.
 * 
 */
struct SquareCell
{
   static constexpr Integer Dim = 2;

   const Real h_x, h_y;
   const Point<Dim> origin;

   GENDIL_HOST_DEVICE
   SquareCell( Point< Dim > origin,
               Real h ) :
      h_x( h ), h_y( h ), origin( origin )
   { }

   GENDIL_HOST_DEVICE
   SquareCell( Point< Dim > origin,
               Real h_x,
               Real h_y ) :
      h_x( h_x ), h_y( h_y ), origin( origin )
   { }

   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::array< Real, Dim >;
   template < typename IntRule >
   using QuadData =  std::tuple<
                        std::tuple_element_t< 0, typename IntRule::points::points_1d_tuple >,
                        std::tuple_element_t< 1, typename IntRule::points::points_1d_tuple >
                     >;

   template < typename QuadData >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( TensorIndex< Dim > const & quad_index,
                              QuadData const & quad_data,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      GetValuesAndJacobian( quad_index[0], quad_index[1], quad_data, X, J_mesh );
   }

   template < typename QuadData >
   GENDIL_HOST_DEVICE
   void GetValuesAndJacobian( const LocalIndex qx,
                              const LocalIndex qy,
                              QuadData const & quad_data,
                              physical_coordinates & X,
                              jacobian & J_mesh ) const
   {
      X[0] = origin[0] + h_x * GetCoord<0>( quad_data, qx );
      X[1] = origin[1] + h_y * GetCoord<1>( quad_data, qy );
      J_mesh[0] = h_x;
      J_mesh[1] = h_y;
   }

   GENDIL_HOST_DEVICE
   jacobian ComputeJacobian( const Point< Dim > & ref_point ) const
   {
      jacobian J_mesh{};
      J_mesh[0] = h_x;
      J_mesh[1] = h_y;
      return J_mesh;
   }
};

GENDIL_HOST_DEVICE
void ApplyOrientationToCell(const Permutation<2>& orientation, SquareCell& cell)
{
   GENDIL_VERIFY( orientation == MakeReferencePermutation<2>(),
      "Orientation of SquareCell must be the reference orientation." );
}

}
