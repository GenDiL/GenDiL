// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Meshes/Geometries/point.hpp"

namespace gendil {

/**
 * @brief A simple structure representing a stretched cubic cell.
 * 
 */
struct CubeCell
{
    static constexpr Integer Dim = 3;

    const Real h_x, h_y, h_z;
    const Point<Dim> origin;

   using physical_coordinates = std::array< Real, Dim >;
   using jacobian = std::tuple< Real, Real, Real >;
    template < typename IntRule >
    using QuadData =  std::tuple<
                        typename std::tuple_element_t<0, typename IntRule::points::points_1d_tuple >,
                        typename std::tuple_element_t<1, typename IntRule::points::points_1d_tuple >,
                        typename std::tuple_element_t<2, typename IntRule::points::points_1d_tuple >
                    >;

   GENDIL_HOST_DEVICE
   CubeCell( Point< Dim > origin,
             Real h ) :
        h_x( h ), h_y( h ), h_z( h ),
        origin( origin )
   { }

   GENDIL_HOST_DEVICE
   CubeCell( Point< Dim > origin,
             Real h_x,
             Real h_y,
             Real h_z ) :
        h_x( h_x ), h_y( h_y ), h_z( h_z ),
        origin( origin )
   { }

    template < typename QuadData >
    GENDIL_HOST_DEVICE
    void GetValuesAndJacobian( TensorIndex< Dim > const & quad_index,
                               QuadData const & quad_data,
                               physical_coordinates & X,
                               jacobian & J_mesh ) const
    {
        GetValuesAndJacobian( quad_index[0], quad_index[1], quad_index[2], quad_data, X, J_mesh );
    }

    template < typename QuadData >
    GENDIL_HOST_DEVICE
    void GetValuesAndJacobian( const LocalIndex qx,
                               const LocalIndex qy,
                               const LocalIndex qz,
                               QuadData const & quad_data,
                               physical_coordinates & X,
                               jacobian & J_mesh ) const
    {
        using quad_x = std::tuple_element_t< 0, QuadData >;
        using quad_y = std::tuple_element_t< 1, QuadData >;
        using quad_z = std::tuple_element_t< 2, QuadData >;
        const auto q_pt_x = quad_x::GetCoord( qx );
        const auto q_pt_y = quad_y::GetCoord( qy );
        const auto q_pt_z = quad_z::GetCoord( qz );
        X[0] = origin[0] + h_x * q_pt_x;
        X[1] = origin[1] + h_y * q_pt_y;
        X[2] = origin[2] + h_z * q_pt_z;
        std::get< 0 >( J_mesh ) = h_x;
        std::get< 1 >( J_mesh ) = h_y;
        std::get< 2 >( J_mesh ) = h_z;
    }
};

}