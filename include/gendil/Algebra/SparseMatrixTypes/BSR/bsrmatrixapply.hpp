// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Shared BSR apply declarations, validation, and dispatch diagnostics.
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixapplycommon.hpp"

// Host BSR matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/BSR/hostbsrmatrixapply.hpp"

// Native CUDA/HIP BSR matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/BSR/nativedevicebsrmatrixapply.hpp"

// cuSPARSE BSR matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/BSR/cusparsebsrmatrixapply.hpp"

// rocSPARSE BSR matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/BSR/rocsparsebsrmatrixapply.hpp"
