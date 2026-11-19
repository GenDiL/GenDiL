// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Meshes/Cells/ReferenceCells/quadcell.hpp>

// Instantiate before the harness can supply missing header dependencies.
template GENDIL_HOST_DEVICE void gendil::ApplyOrientationToCell<3>(
   const gendil::Permutation<2>&, gendil::QuadCell<3>&);

#include "oriented-cell-header-test.hpp"

int main()
{
   return oriented_cell_header_test::Run<gendil::QuadCell<3>>();
}
