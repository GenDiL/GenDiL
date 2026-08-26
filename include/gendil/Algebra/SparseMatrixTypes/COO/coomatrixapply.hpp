// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Shared COO apply declarations, validation, and dispatch diagnostics.
#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixapplycommon.hpp"

// Host COO matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/COO/hostcoomatrixapply.hpp"

// Native CUDA/HIP COO matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/COO/nativedevicecoomatrixapply.hpp"

// cuSPARSE COO matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/COO/cusparsecoomatrixapply.hpp"

// rocSPARSE COO matrix-vector product implementation.
#include "gendil/Algebra/SparseMatrixTypes/COO/rocsparsecoomatrixapply.hpp"
