// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Generic algorithms to assemble finite element matrices from a weak-form expression.
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/genericassembly.hpp"

// Matrix assembly/operator factory selector.
#include "gendil/FiniteElementMethod/MatrixAssembly/matrixassemblytype.hpp"

// Assembly of block sparse matrices in BSR format.
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrassembly.hpp"

// Scatter/gather BSR assembly wrapper for finite-element vectors.
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrassembly.hpp"

// Raw COO triplet sparse assembly (unsorted and duplicated entries).
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/cooassembly.hpp"

