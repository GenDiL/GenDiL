// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

// Public selector for matrix assembly/operator factories. BSR returns the raw
// element-block sparse matrix, while SGBSR wraps that element-block operator in
// finite-element gather/scatter maps.
enum class MatrixAssemblyType
{
   BSR,
   SGBSR,
   RawCOO,
   COO,
   CSR,
   CSC
};

} // namespace gendil
