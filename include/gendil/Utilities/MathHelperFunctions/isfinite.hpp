// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

#include <limits>

namespace gendil
{

/// Return whether a real scalar is neither NaN nor infinite.
GENDIL_HOST_DEVICE inline bool IsFinite(const Real value)
{
   constexpr Real largest = std::numeric_limits<Real>::max();
   return value >= -largest && value <= largest;
}

} // namespace gendil
