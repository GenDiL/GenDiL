// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include <type_traits>
#include <utility>

namespace gendil
{
namespace details
{
   struct KernelLaunchGeometry
   {
      GlobalIndex grid_x;
      GlobalIndex block_x;
      GlobalIndex block_y;
      GlobalIndex block_z;
   };

   GENDIL_HOST_DEVICE
   constexpr GlobalIndex CeilDiv(
      const GlobalIndex n,
      const GlobalIndex divisor )
   {
      return divisor == 0 ? 0 : ( n + divisor - 1 ) / divisor;
   }

#if defined( GENDIL_USE_DEVICE )
   template < typename KernelConfiguration, typename Lambda >
   __global__ static
   void BlockLoopKernel( const GlobalIndex num_work_items, Lambda body )
   {
      if ( KernelConfiguration::IsActive( num_work_items ) )
      {
         body( KernelConfiguration::WorkItemIndex() );
      }
   }

   /**
    * @brief Diagnostic-only all-candidate kernel.
    *
    * @details CandidateBlockLoop users see every physical candidate lane,
    * including inactive lanes in the final partial batch. The body takes no
    * argument; diagnostics should query lane metadata through static
    * KernelConfiguration accessors such as BatchIndex(), WorkItemIndex(), and
    * IsActive(n). Production code must use BlockLoop instead.
    */
   template < typename KernelConfiguration, typename Lambda >
   __global__ static
   void CandidateBlockLoopKernel( const GlobalIndex num_work_items, Lambda body )
   {
      (void) num_work_items;
      body();
   }
#endif

   template < typename KernelConfiguration >
   struct shared_memory_stride
   {
      static constexpr size_t value( const size_t per_work_item_reals )
      {
         if constexpr ( requires {
            KernelConfiguration::SharedMemoryStride( per_work_item_reals );
         } )
         {
            return KernelConfiguration::SharedMemoryStride(
               per_work_item_reals );
         }
         else
         {
            return per_work_item_reals;
         }
      }
   };

   template < typename KernelConfiguration >
   struct shared_memory_block_size
   {
      static constexpr size_t value( const size_t per_work_item_reals )
      {
         if constexpr ( requires {
            KernelConfiguration::SharedMemoryBlockSize( per_work_item_reals );
         } )
         {
            return KernelConfiguration::SharedMemoryBlockSize(
               per_work_item_reals );
         }
         else
         {
            return per_work_item_reals;
         }
      }
   };

   template < typename KernelConfiguration >
   GENDIL_HOST_DEVICE
   inline Real * SharedMemoryForConfiguration(
      Real * shared_data,
      const size_t per_work_item_reals )
   {
      if constexpr ( requires {
         KernelConfiguration::SharedMemoryForWorkItem(
            shared_data,
            per_work_item_reals );
      } )
      {
         return KernelConfiguration::SharedMemoryForWorkItem(
            shared_data,
            per_work_item_reals );
      }
      else
      {
         return shared_data;
      }
   }
} // namespace details
} // namespace gendil
