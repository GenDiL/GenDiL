// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/MathHelperFunctions/isfinite.hpp"
#include "gendil/Utilities/MathHelperFunctions/sqrt.hpp"

#include <limits>

namespace gendil
{

/// Return the square root of a finite nonnegative value, or NaN otherwise.
GENDIL_HOST_DEVICE inline Real CheckedSqrt(const Real value)
{
   return IsFinite(value) && value >= 0.0
      ? Sqrt(value) : std::numeric_limits<Real>::quiet_NaN();
}

} // namespace gendil
