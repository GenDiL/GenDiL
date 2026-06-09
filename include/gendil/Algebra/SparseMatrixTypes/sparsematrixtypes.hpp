// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// A block sparse row (BSR) sparse matrix type.
#include "gendil/Algebra/SparseMatrixTypes/bsrmatrix.hpp"

// Algebraic scatter/gather wrapper for BSR sparse matrix application.
#include "gendil/Algebra/SparseMatrixTypes/sgbsrmatrix.hpp"

// Raw duplicated unsorted COO triplet assembly buffer.
#include "gendil/Algebra/SparseMatrixTypes/rawcootripletbuffer.hpp"

// Canonical sorted and duplicate-reduced COO sparse matrix type.
#include "gendil/Algebra/SparseMatrixTypes/coomatrix.hpp"

// Canonical compressed sparse row sparse matrix type.
#include "gendil/Algebra/SparseMatrixTypes/csrmatrix.hpp"

// Canonical compressed sparse column sparse matrix type.
#include "gendil/Algebra/SparseMatrixTypes/cscmatrix.hpp"
