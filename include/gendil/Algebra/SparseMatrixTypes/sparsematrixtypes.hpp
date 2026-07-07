// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Shared sparse matrix action arithmetic policy helpers.
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"

// Marker bases for host/device sparse matrix-vector application backends.
#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"

// A block sparse row (BSR) sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrix.hpp"

// Algebraic scatter/gather wrapper for BSR sparse matrix application.
#include "gendil/Algebra/SparseMatrixTypes/SGBSR/sgbsrmatrix.hpp"

// Raw duplicated unsorted COO triplet assembly buffer.
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcootripletbuffer.hpp"

// Canonical sorted and duplicate-reduced COO sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrix.hpp"

// Canonical compressed sparse row sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrix.hpp"

// Canonical compressed sparse column sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrix.hpp"

#ifdef GENDIL_USE_HYPRE
// Hypre-compatible compressed sparse row sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/HypreCSR/hyprecsrmatrix.hpp"
#endif
