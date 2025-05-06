// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/KernelContext/KernelConfigurations/threadfirst.hpp"

namespace gendil
{

using SerialKernelConfiguration = ThreadFirstKernelConfiguration< ThreadBlockLayout<>, 0 >;

}