// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_HYPRE

// Move-only Hypre-compatible CSR storage and zero-copy shell integration.
#include "gendil/Algebra/SparseMatrixTypes/HypreCSR/hyprecsrmatrixstorage.hpp"

// Hypre-native and explicitly forwarded CSR Apply support.
#include "gendil/Algebra/SparseMatrixTypes/HypreCSR/hyprecsrmatrixapply.hpp"

#endif // GENDIL_USE_HYPRE
