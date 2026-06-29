// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
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
      return HyperCube< Dim >::GetNormalDimensionIndex( face_index );
   }

   static constexpr int GetNormalSign( Integer face_index )
   {
      return HyperCube< Dim >::GetNormalSign( face_index );
   }

   GENDIL_HOST_DEVICE
   jacobian ComputeJacobian( const Point< Dim > & ref_point ) const
   {
      jacobian J_mesh{};
      for (Integer i = 0; i < Dim; ++i)
      {
            J_mesh[i] = h[i];
      }
      return J_mesh;
   }
};

template < Integer Dim >
GENDIL_HOST_DEVICE
void ApplyOrientationToCell( const Permutation<Dim>& orientation, HyperCubeCell<Dim>& cell )
{
   GENDIL_VERIFY( orientation == MakeReferencePermutation<Dim>(),
      "Orientation of HyperCubeCell must be the reference orientation." );
}

template < typename Orientation, Integer Dim >
GENDIL_HOST_DEVICE
void ApplyOrientationToCell( const Orientation& orientation, HyperCubeCell<Dim>& cell )
{
   // GENDIL_VERIFY( orientation == MakeReferencePermutation<Dim>(),
   //    "Orientation of HyperCubeCell must be the reference orientation." );
}

}
