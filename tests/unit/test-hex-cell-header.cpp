// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Meshes/Cells/ReferenceCells/hexcell.hpp>

// Instantiate before the harness can supply missing header dependencies.
template GENDIL_HOST_DEVICE void gendil::ApplyOrientationToCell<3>(
   const gendil::Permutation<3>&, gendil::HexCell<3>&);

#include "oriented-cell-header-test.hpp"

int main()
{
   return oriented_cell_header_test::Run<gendil::HexCell<3>>();
}
