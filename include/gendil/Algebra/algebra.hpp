// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Componentwise addition of fixed-size arrays and mixed array storage.
#include "gendil/Algebra/arrayaddition.hpp"

// A concept for statically sized vectors.
#include "gendil/Algebra/staticvector.hpp"

// Accessor helpers for vector and matrix indexing.
#include "gendil/Algebra/accessors.hpp"

// Matrix-vector multiplication operations.
#include "gendil/Algebra/matrixvectorops.hpp"

// A simple vector type.
#include "gendil/Algebra/Vector/vector.hpp"

// Unified host/device access for GenDiL and external vector types.
#include "gendil/Algebra/Vector/access.hpp"

// Borrow equally sized slices of a contiguous vector without copying entries.
#include "gendil/Algebra/Vector/slices.hpp"

// BLAS-like operations on owning vectors and generic vector views.
#include "gendil/Algebra/Vector/operations.hpp"

// Completed vector operations inside a native kernel context.
#include "gendil/Algebra/Vector/localoperations.hpp"

// Explicit Euclidean reductions, projections, and fused basis updates.
#include "gendil/Algebra/Vector/reductions.hpp"

// Callable protocol for sized maps between vectors.
#include "gendil/Algebra/vectormapconcept.hpp"

// Allocation-free heterogeneous tuple composition of rectangular vector maps.
#include "gendil/Algebra/blockvectormap.hpp"

// The supported sparse matrix types.
#include "gendil/Algebra/SparseMatrixTypes/sparsematrixtypes.hpp"
