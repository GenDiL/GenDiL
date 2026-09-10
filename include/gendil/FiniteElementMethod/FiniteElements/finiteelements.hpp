// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Public finite-element representation and composition facilities.
 *
 * Include the narrower headers below when only one construction family is
 * needed, or include this umbrella for the complete finite-element API.
 */

// Reference-geometry and shape-function pairing.
#include "gendil/FiniteElementMethod/FiniteElements/finiteelement.hpp"

// Scalar tensor-product Legendre and Lobatto finite elements.
#include "gendil/FiniteElementMethod/FiniteElements/lagrangefiniteelement.hpp"

// Heterogeneous and homogeneous component-vector finite elements.
#include "gendil/FiniteElementMethod/FiniteElements/vectorfiniteelement.hpp"

// Tensor composition with scalar broadcast and componentwise vector zip.
#include "gendil/FiniteElementMethod/FiniteElements/tensorfiniteelement.hpp"
