// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorindex.hpp"

namespace gendil {

namespace mesh {

template <
   typename Cell,
   typename QuadData
>
GENDIL_HOST_DEVICE
void GetValuesAndJacobian(
   const Cell & cell,
   TensorIndex< Cell::Dim > const & quad_index,
   QuadData const & quad_data,
   typename Cell::physical_coordinates & X,
   typename Cell::jacobian & J_mesh )
{
   cell.GetValuesAndJacobian( quad_index, quad_data, X, J_mesh );
}
template <
   typename Cell,
   CellFaceView Face,
   typename FaceQuadData
>
GENDIL_HOST_DEVICE
void ComputePhysicalCoordinatesAndJacobian(
   const Cell & cell,
   const Face & face,
   TensorIndex< Cell::Dim > const & quad_index,
   FaceQuadData const & face_quad_data,
   typename Cell::physical_coordinates & X,
   typename Cell::jacobian & J_mesh )
{
   constexpr Integer local_face_index = Face::local_face_index_type::value;
   const auto & local_face_quad_data = std::get< local_face_index >( face_quad_data );
   cell.GetValuesAndJacobian( quad_index, local_face_quad_data, X, J_mesh );
}

}

}