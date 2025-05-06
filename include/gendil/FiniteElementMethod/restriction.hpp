// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

struct H1Restriction
{
   HostDevicePointer< const int > indices;
   const Integer num_dofs;
};

struct L2Restriction{};

}