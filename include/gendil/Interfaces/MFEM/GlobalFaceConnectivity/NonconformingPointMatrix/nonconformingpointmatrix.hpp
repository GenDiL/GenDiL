// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Complete internal MFEM nonconforming point-matrix decoder module.
 *
 * The decoder uses copied MFEM point-matrix data, performs no allocation, and
 * does not depend on the MFEM runtime. It is an internal GenDiL MFEM-interface
 * implementation detail; production builders and tests should include this
 * umbrella rather than the facade or algorithm subheaders directly.
 *
 * Decoding uses a bounded exhaustive orientation search per nonconforming
 * leaf. This cost is paid during connectivity construction only; no
 * point-matrix decode work is performed during operator execution.
 */

#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/NonconformingPointMatrix/nonconformingpointmatrixdecoder.hpp"
#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/NonconformingPointMatrix/nonconformingpointmatrixdecodealgorithm.hpp"
