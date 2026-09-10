// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// A concept for statically sized vectors.
#include "gendil/Algebra/staticvector.hpp"

// Accessor helpers for vector and matrix indexing.
#include "gendil/Algebra/accessors.hpp"

// Matrix-vector multiplication operations.
#include "gendil/Algebra/matrixvectorops.hpp"

// A simple vector type.
#include "gendil/Algebra/vector.hpp"

// Unified host/device access for GenDiL and external vector types.
#include "gendil/Algebra/vectoraccess.hpp"

// BLAS-like operations on owning vectors and generic vector views.
#include "gendil/Algebra/vectoroperations.hpp"

// Callable protocol for sized maps between vectors.
#include "gendil/Algebra/vectormapconcept.hpp"

// Allocation-free heterogeneous tuple composition of rectangular vector maps.
#include "gendil/Algebra/blockvectormap.hpp"

// The supported sparse matrix types.
#include "gendil/Algebra/SparseMatrixTypes/sparsematrixtypes.hpp"
