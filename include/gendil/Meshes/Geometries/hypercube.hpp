// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MathHelperFunctions/pow.hpp"
// #include "tensorgeometry.hpp"

namespace gendil {

/**
 * @brief A structure describing an hypercube geometry.
 * 
 * @tparam GeometryDim The dimension of the hypercube.
 * @tparam SpaceDim The dimension of the space embedding the geometry.
 */
template < Integer GeometryDim, Integer SpaceDim = GeometryDim >
struct HyperCube //: public TensorGeometry< Segment< 1 >, HyperCube< Dim - 1 > >
{
   static constexpr Integer space_dim = SpaceDim;
   static constexpr Integer geometry_dim = GeometryDim;
   static constexpr Integer num_nodes = Pow< GeometryDim >( 2 );
   static constexpr Integer num_faces = 2 * GeometryDim;

   Point< space_dim > vertices[ num_nodes ];

   // using faces_type = std::tuple< Point< space_dim >, Point< space_dim > >;
   // faces_type faces;

   template < typename... Points >
   constexpr HyperCube( Points const & ... points ) :
      vertices{ points... }//, faces{ point_A, point_B }
   { }
};

// template < >
// struct HyperCube< 1 > : public Segment< 1 >
// {
// };

}
