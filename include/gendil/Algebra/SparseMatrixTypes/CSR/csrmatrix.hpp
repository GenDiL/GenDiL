// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Move-only CSR matrix storage and backend definitions.
#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixstorage.hpp"

// Raw-pointer CSR views and synchronized host/device access.
#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixview.hpp"

// Host, native-device, cuSPARSE, and rocSPARSE CSR Apply support.
#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixapply.hpp"
