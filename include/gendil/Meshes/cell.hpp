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
void GetValuesAndJacobian(
   const Cell & cell,
   TensorIndex< Cell::Dim > const & quad_index,
   QuadData const & quad_data,
   typename Cell::physical_coordinates & X,
   typename Cell::jacobian & J_mesh )
{
   cell.GetValuesAndJacobian( quad_index, quad_data, X, J_mesh );
}

}

}