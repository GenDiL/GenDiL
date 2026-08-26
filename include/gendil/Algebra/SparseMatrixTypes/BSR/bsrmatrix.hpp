// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// BSR backend configuration customization point.
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrbackendconfiguration.hpp"

// Move-only BSR matrix storage and backend definitions.
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixstorage.hpp"

// Raw-pointer BSR views and synchronized host/device access.
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixview.hpp"

// Host, native-device, cuSPARSE, and rocSPARSE BSR Apply support.
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixapply.hpp"
