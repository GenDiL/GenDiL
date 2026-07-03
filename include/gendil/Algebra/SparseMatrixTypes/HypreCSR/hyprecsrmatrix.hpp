// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_HYPRE

// Public Hypre-compatible CSR matrix interface: storage plus native Apply support.
#include "gendil/Algebra/SparseMatrixTypes/HypreCSR/hyprecsrmatrixstorage.hpp"
#include "gendil/Algebra/SparseMatrixTypes/HypreCSR/hyprecsrmatrixapply.hpp"

#endif // GENDIL_USE_HYPRE
