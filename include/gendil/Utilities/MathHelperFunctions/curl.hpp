// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

GENDIL_HOST_DEVICE
Real Curl(const Real (& J)[1][1]) {
   return 0.0;
}

GENDIL_HOST_DEVICE
Real Curl(const Real (& J)[2][2]) {
   return J[1][0] - J[0][1];
}

GENDIL_HOST_DEVICE
std::array<Real,3> Curl(const Real (& J)[3][3]) {
   return {
      J[2][1] - J[1][2],  // curl_x
      J[0][2] - J[2][0],  // curl_y
      J[1][0] - J[0][1]   // curl_z
   };
}

}
