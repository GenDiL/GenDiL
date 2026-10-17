// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

// Public selector for matrix assembly/operator factories. BSR returns the raw
// element-block sparse matrix, SGBSR wraps that element-block operator in
// finite-element gather/scatter maps, RawCOO returns the intermediate triplet
// stream, and COO/CSR/CSC return canonical sparse matrix storage.
enum class MatrixAssemblyType
{
   BSR,
   SGBSR,
   RawCOO,
   COO,
   CSR,
   CSC
#ifdef GENDIL_USE_HYPRE
   ,
   HypreCSR
#endif
};

// Select all element couplings or retain only element self-couplings.
// This controls which contributions are assembled, independently of storage format.
enum class SparseAssemblyMode
{
   Full,
   ElementBlockDiagonal
};

} // namespace gendil
