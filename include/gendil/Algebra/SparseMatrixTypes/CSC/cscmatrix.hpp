// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Move-only CSC matrix storage and backend definitions.
#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixstorage.hpp"

// Raw-pointer CSC views and synchronized host/device access.
#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixview.hpp"

// Host, native-device, cuSPARSE, and rocSPARSE CSC Apply support.
#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixapply.hpp"
