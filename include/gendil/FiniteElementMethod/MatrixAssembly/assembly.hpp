// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Select the matrix storage format and full or element-block assembly.
#include "gendil/FiniteElementMethod/MatrixAssembly/matrixassemblytype.hpp"

// Generic algorithms to assemble finite element matrices from a weak-form expression.
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/genericassembly.hpp"

// Shared scratch requirements for local sparse-assembly probes.
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/requiredsharedmemory.hpp"

// Assembly of block sparse matrices in BSR format.
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrassembly.hpp"

// Scatter/gather BSR assembly wrapper for finite-element vectors.
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrassembly.hpp"

// Raw COO triplet sparse assembly (unsorted and duplicated entries).
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/cooassembly.hpp"

// CSR sparse assembly through RawCOO host sort/reduce finalization.
#include "gendil/FiniteElementMethod/MatrixAssembly/CSR/csrassembly.hpp"

// CSC sparse assembly through RawCOO host sort/reduce finalization.
#include "gendil/FiniteElementMethod/MatrixAssembly/CSC/cscassembly.hpp"

#ifdef GENDIL_USE_HYPRE
// Hypre-compatible CSR sparse assembly through RawCOO host sort/reduce finalization.
#include "gendil/FiniteElementMethod/MatrixAssembly/HypreCSR/hyprecsrassembly.hpp"
#endif
