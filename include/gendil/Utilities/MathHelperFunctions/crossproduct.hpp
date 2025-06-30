// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

// 1D: trivial zero
GENDIL_HOST_DEVICE
Real Cross(const std::array<Real,1> &a, const std::array<Real,1> &b) {
    return 0.0;
}

// 2D: scalar pseudocross => return the scalar pseudoscalar
GENDIL_HOST_DEVICE
Real Cross(const std::array<Real,2> &a, const std::array<Real,2> &b) {
    return a[0]*b[1] - a[1]*b[0];
}

// 3D: standard cross product
GENDIL_HOST_DEVICE
std::array<Real,3> Cross(const std::array<Real,3> &a, const std::array<Real,3> &b) {
    return {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}

}
