// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Shared sparse matrix action arithmetic policy helpers.
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"

// Marker bases for host/device sparse matrix-vector application backends.
#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"

// A block sparse row (BSR) sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/bsrmatrix.hpp"

// Algebraic scatter/gather wrapper for BSR sparse matrix application.
#include "gendil/Algebra/SparseMatrixTypes/sgbsrmatrix.hpp"

// Raw duplicated unsorted COO triplet assembly buffer.
#include "gendil/Algebra/SparseMatrixTypes/rawcootripletbuffer.hpp"

// Canonical sorted and duplicate-reduced COO sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/coomatrix.hpp"

// Canonical compressed sparse row sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/csrmatrix.hpp"

// Canonical compressed sparse column sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/cscmatrix.hpp"

#ifdef GENDIL_USE_HYPRE
// Hypre-compatible compressed sparse row sparse matrix type and action.
#include "gendil/Algebra/SparseMatrixTypes/hyprecsrmatrix.hpp"
#endif
