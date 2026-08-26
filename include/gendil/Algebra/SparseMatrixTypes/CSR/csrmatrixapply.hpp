// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Shared CSR apply declarations, validation, and dispatch diagnostics.
#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixapplycommon.hpp"

// Host CSR matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/CSR/hostcsrmatrixapply.hpp"

// Native CUDA/HIP CSR matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/CSR/nativedevicecsrmatrixapply.hpp"

// cuSPARSE CSR matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/CSR/cusparsecsrmatrixapply.hpp"

// rocSPARSE CSR matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/CSR/rocsparsecsrmatrixapply.hpp"
