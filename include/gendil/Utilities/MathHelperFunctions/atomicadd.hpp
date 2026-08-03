// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

/// Atomically add a value and return the value observed before the update.
template <typename T>
GENDIL_HOST_DEVICE T AtomicAdd(T &add, const T val)
{
#ifdef GENDIL_DEVICE_CODE
   return atomicAdd(&add,val);
#else
   T old;
   #pragma omp atomic capture
   {
      old = add;
      add += val;
   }
   return old;
#endif
}

/// Atomically add a value without returning the value preceding the update.
template <typename T>
GENDIL_HOST_DEVICE void AtomicAddInPlace(T &add, const T val)
{
#ifdef GENDIL_DEVICE_CODE
   (void) atomicAdd(&add,val);
#else
   #pragma omp atomic update
   add += val;
#endif
}

}
