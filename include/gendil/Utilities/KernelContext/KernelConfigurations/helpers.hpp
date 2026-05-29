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

   template < typename KernelConfiguration, typename Lambda >
   GENDIL_HOST_DEVICE
   inline void InvokeDeviceKernelBody(
      const KernelConfiguration & kernel_configuration,
      const GlobalIndex num_work_items,
      Lambda & body )
   {
      if constexpr ( std::is_invocable_v< Lambda, KernelConfiguration > )
      {
         body( kernel_configuration );
      }
      else if constexpr ( std::is_invocable_v< Lambda, GlobalIndex > )
      {
         static_assert(
            KernelConfiguration::batch_size == 1,
            "One-index kernel bodies are only safe for BatchSize == 1. "
            "Use a kernel-configuration body and guard reads, writes, and "
            "atomics with IsActive(num_work_items) for batched execution." );

         if ( kernel_configuration.IsActive( num_work_items ) )
         {
            body( kernel_configuration.WorkItemIndex() );
         }
      }
      else
      {
         static_assert(
            std::is_invocable_v< Lambda, KernelConfiguration > ||
            std::is_invocable_v< Lambda, GlobalIndex >,
            "Kernel body must be invocable with either a kernel configuration "
            "object or a GlobalIndex work-item index." );
      }
   }

#if defined( GENDIL_USE_DEVICE )
   template < typename KernelConfiguration, typename Lambda >
   __global__ static
   void DeviceGridLoop( const GlobalIndex num_work_items, Lambda body )
   {
      KernelConfiguration kernel_configuration;
      InvokeDeviceKernelBody(
         kernel_configuration,
         num_work_items,
         body );
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
      const KernelConfiguration & kernel_configuration,
      Real * shared_data,
      const size_t per_work_item_reals )
   {
      if constexpr ( requires {
         kernel_configuration.SharedMemoryForWorkItem(
            shared_data,
            per_work_item_reals );
      } )
      {
         return kernel_configuration.SharedMemoryForWorkItem(
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
