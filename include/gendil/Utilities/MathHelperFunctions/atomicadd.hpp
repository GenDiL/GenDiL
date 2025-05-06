// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

template <typename T>
GENDIL_HOST_DEVICE T AtomicAdd(T &add, const T val)
{
#ifdef GENDIL_DEVICE_CODE
   return atomicAdd(&add,val);
#else
   T old = add;
   #pragma omp atomic
   add += val;
   return old;
#endif
}

}
