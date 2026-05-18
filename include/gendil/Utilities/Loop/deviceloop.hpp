// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil {

namespace details
{
#ifdef GENDIL_USE_DEVICE
   template <typename Lambda> __global__ static
   void GPULoop(const int n, Lambda body)
   {
      const int tid =
         blockIdx.x * blockDim.x + threadIdx.x;

      const int stride =
         blockDim.x * gridDim.x;

      for (int k = tid; k < n; k += stride)
      {
         body(k);
      }
   }
#endif
} // namespace details

template <typename Lambda, int block_size = 256>
void DeviceLoop( const int n, Lambda && body )
{
#ifdef GENDIL_USE_DEVICE
   const int num_blocks = (n + block_size - 1) / block_size;
   GENDIL_CHECK_NO_PENDING_DEVICE_ERROR("DeviceLoop: before launch");
   details::GPULoop<<<num_blocks, block_size>>>( n, body );
   GENDIL_CHECK_LAST_DEVICE_LAUNCH("DeviceLoop");
#else
   #pragma omp parallel for
   for ( int k = 0; k < n; ++k ) { body( k ); }
#endif
}

} // namespace gendil
