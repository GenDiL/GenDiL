// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/Loop/deviceloop.hpp"

#include <type_traits>
#include <utility>

namespace gendil
{

/**
 * Execute one loop iteration per index in the selected kernel memory space.
 *
 * Device execution delegates to DeviceLoop. Host execution uses OpenMP when
 * enabled and otherwise remains a serial loop.
 */
template < bool OnDevice, typename IndexType, typename Lambda >
void KernelLoop( const IndexType n, Lambda && body )
{
   static_assert(
      std::is_integral_v< IndexType > &&
      !std::is_same_v< std::remove_cv_t< IndexType >, bool >,
      "KernelLoop requires an integral iteration count." );

   if constexpr ( OnDevice )
   {
      DeviceLoop( n, std::forward< Lambda >( body ) );
   }
   else
   {
      #pragma omp parallel for
      for ( IndexType k = 0; k < n; ++k )
      {
         body( k );
      }
   }
}

} // namespace gendil
