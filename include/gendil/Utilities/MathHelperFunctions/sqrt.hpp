// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

#include <cmath>

namespace gendil {

GENDIL_HOST_DEVICE inline
Real Sqrt( Real val )
{
   return std::sqrt( val );
}

}
