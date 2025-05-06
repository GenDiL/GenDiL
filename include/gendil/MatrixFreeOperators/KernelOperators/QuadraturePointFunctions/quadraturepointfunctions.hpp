// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Include math functions
#include "gendil/Utilities/MathHelperFunctions/mathhelperfunctions.hpp"

// Compute the quadrature weight from DofToQuad tuple.
#include "getweight.hpp"

// Apply mapping from reference to physical space.
#include "applymapping.hpp"

// Apply transposed mapping from reference to physical space.
#include "applymappingtranspose.hpp"

// Compute the physical normal from the jacobian and reference normal.
#include "computephysicalnormal.hpp"

// Aggregate dimensions of a tensor.
#include "aggregatedimensions.hpp"
