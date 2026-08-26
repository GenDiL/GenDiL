// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/debug.hpp"

#include <limits>
#include <type_traits>

namespace gendil {

namespace details
{
#ifdef GENDIL_USE_DEVICE
   template < typename IndexType, typename Lambda > __global__ static
   void GPULoop( const IndexType n, Lambda body )
   {
      const IndexType tid =
         static_cast< IndexType >( blockIdx.x ) *
         static_cast< IndexType >( blockDim.x ) +
         static_cast< IndexType >( threadIdx.x );

      const IndexType stride =
         static_cast< IndexType >( blockDim.x ) *
         static_cast< IndexType >( gridDim.x );

      for ( IndexType k = tid; k < n; k += stride )
      {
         body( k );
      }
   }
#endif
} // namespace details

template <
   typename IndexType,
   typename Lambda,
   unsigned int block_size = 256 >
void DeviceLoop( const IndexType n, Lambda && body )
{
   static_assert(
      std::is_integral_v< IndexType > &&
      !std::is_same_v< std::remove_cv_t< IndexType >, bool >,
      "DeviceLoop requires an integral iteration count." );
   static_assert( block_size > 0, "DeviceLoop requires a positive block size." );

   if constexpr ( std::is_signed_v< IndexType > )
   {
      if ( n <= 0 )
      {
         return;
      }
   }
   else if ( n == 0 )
   {
      return;
   }

#ifdef GENDIL_USE_DEVICE
   using UnsignedIndexType = std::make_unsigned_t< IndexType >;
   const UnsignedIndexType count =
      static_cast< UnsignedIndexType >( n );
   const UnsignedIndexType num_blocks =
      UnsignedIndexType( 1 ) +
      ( count - UnsignedIndexType( 1 ) ) /
         static_cast< UnsignedIndexType >( block_size );

   GENDIL_VERIFY(
      num_blocks <=
         static_cast< UnsignedIndexType >(
            std::numeric_limits< unsigned int >::max() ),
      "DeviceLoop launch grid is too large." );

   const dim3 grid_dim( static_cast< unsigned int >( num_blocks ) );
   const dim3 block_dim( block_size );
   CheckDeviceLaunchConfiguration( grid_dim, block_dim, 0 );
   GENDIL_CHECK_NO_PENDING_DEVICE_ERROR("DeviceLoop: before launch");
   details::GPULoop<<< grid_dim, block_dim >>>( n, body );
   GENDIL_CHECK_LAST_DEVICE_LAUNCH("DeviceLoop");
#else
   #pragma omp parallel for
   for ( IndexType k = 0; k < n; ++k ) { body( k ); }
#endif
}

} // namespace gendil
