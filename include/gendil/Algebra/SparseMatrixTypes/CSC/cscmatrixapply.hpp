// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Shared CSC apply declarations, validation, and dispatch diagnostics.
#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixapplycommon.hpp"

// Host CSC matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/CSC/hostcscmatrixapply.hpp"

// Native CUDA/HIP CSC matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/CSC/nativedevicecscmatrixapply.hpp"

// cuSPARSE CSC matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/CSC/cusparsecscmatrixapply.hpp"

// rocSPARSE CSC matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/CSC/rocsparsecscmatrixapply.hpp"
