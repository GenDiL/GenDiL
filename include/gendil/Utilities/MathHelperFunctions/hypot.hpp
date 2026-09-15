// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cmath>
#include "gendil/prelude.hpp"

namespace gendil {

/** @brief Compute sqrt(x*x + y*y) without avoidable overflow or underflow. */
GENDIL_HOST_DEVICE inline
Real Hypot(const Real x, const Real y)
{
#ifdef GENDIL_DEVICE_CODE
   return ::hypot(x, y);
#else
   return std::hypot(x, y);
#endif
}

} // namespace gendil
