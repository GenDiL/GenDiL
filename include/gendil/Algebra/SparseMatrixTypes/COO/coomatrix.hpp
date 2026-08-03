// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Move-only COO matrix storage and backend definitions.
#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixstorage.hpp"

// Raw-pointer COO views and synchronized host/device access.
#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixview.hpp"

// Host, native-device, cuSPARSE, and rocSPARSE COO Apply support.
#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixapply.hpp"
