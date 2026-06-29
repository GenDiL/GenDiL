// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief MFEM global-face connectivity builders.
 *
 * Public MFEM global-face APIs are available only when GENDIL_USE_MFEM is
 * enabled. The current supported surface builds serial, non-embedded
 * Line/Quad/Hex conforming interior faces, Quad/Hex local nonconforming leaf
 * construction, and one-sided boundary faces.
 */

#ifdef GENDIL_USE_MFEM

// Metadata records, options, and bundle return types for MFEM global-face builders.
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/globalfacemetadata.hpp"

// Serial conforming MFEM interior-face connectivity construction.
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/conformingglobalfacebuilder.hpp"

// Serial MFEM nonconforming leaf-face connectivity construction from point matrices.
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/nonconformingglobalfacebuilder.hpp"

// Serial one-sided MFEM boundary-face connectivity construction and filtering.
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/boundaryglobalfacebuilder.hpp"

#endif // GENDIL_USE_MFEM
