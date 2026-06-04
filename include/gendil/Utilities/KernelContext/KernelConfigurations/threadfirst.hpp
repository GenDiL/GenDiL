// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/IndexSequenceHelperFunctions/print.hpp"
#include "gendil/Utilities/KernelContext/KernelConfigurations/helpers.hpp"
#include "gendil/Utilities/KernelContext/threadlayout.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/debug.hpp"

#if defined( GENDIL_USE_MFEM )
#include <mfem.hpp>
#include "general/forall.hpp"
#endif

namespace gendil {

/**
 * @brief Legacy GPU kernel configuration using one semantic work item per
 * thread block.
 *
 * @details This configuration intentionally preserves the legacy physical block
 * shape for comparison with the new batched DeviceKernelConfiguration:
 *  - BatchSize is fixed to 1.
 *  - BatchIndex() is always 0.
 *  - WorkItemIndex() is blockIdx.x on device.
 *  - ThreadBlockLayout< D0, D1, D2 > launches as dim3(D0, D1, D2).
 *  - Higher-dimensional ThreadBlockLayout instances are linearized in
 *    threadIdx.x.
 */
template <
   typename ThreadLayout,
   size_t MaxSharedDimensions >
class ThreadFirstKernelConfiguration
{
public:
   static_assert(
      MaxSharedDimensions >= ThreadLayout::thread_block_dim,
      "ThreadFirstKernelConfiguration requires MaxSharedDimensions to be "
      "greater than or equal to the number of threaded dimensions." );

   using thread_layout_type = ThreadLayout;

   static constexpr bool is_host_configuration = false;
   static constexpr bool is_device_configuration = true;
   static constexpr size_t batch_size = 1;
   static constexpr size_t thread_block_dim = ThreadLayout::thread_block_dim;
   static constexpr size_t shared_block_max_dim = MaxSharedDimensions;

   template < Integer space_dim >
   using threaded_dimensions = typename HeadTailSplit< Min( ThreadLayout::thread_block_dim, space_dim ), std::make_index_sequence< space_dim > >::Head;
   template < Integer space_dim >
   using register_dimensions = typename HeadTailSplit< Min( ThreadLayout::thread_block_dim, space_dim ), std::make_index_sequence< space_dim > >::Tail;

   template < Integer space_dim >
   using non_shared_register_dimensions = typename HeadTailSplit< Min( shared_block_max_dim, space_dim )-Min( ThreadLayout::thread_block_dim, space_dim ), register_dimensions< space_dim > >::Tail;
   template < Integer space_dim >
   using shared_register_dimensions = typename HeadTailSplit< Min( shared_block_max_dim, space_dim )-Min( ThreadLayout::thread_block_dim, space_dim ), register_dimensions< space_dim > >::Head;

   template < Integer space_dim >
   using shared_dimensions = cat_t< threaded_dimensions< space_dim >, shared_register_dimensions< space_dim > >;

   ThreadFirstKernelConfiguration() = default;

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return ThreadLayout::GetNumberOfThreads();
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      return ThreadLayout::template GetBlockDim< Index >();
   }

   GENDIL_HOST_DEVICE
   static constexpr details::KernelLaunchGeometry GetLaunchGeometry(
      const GlobalIndex num_work_items )
   {
      if constexpr ( ThreadLayout::thread_block_dim == 0 )
      {
         return { num_work_items, 1, 1, 1 };
      }
      else if constexpr ( ThreadLayout::thread_block_dim == 1 )
      {
         return {
            num_work_items,
            ThreadLayout::template GetBlockDim< 0 >(),
            1,
            1 };
      }
      else if constexpr ( ThreadLayout::thread_block_dim == 2 )
      {
         return {
            num_work_items,
            ThreadLayout::template GetBlockDim< 0 >(),
            ThreadLayout::template GetBlockDim< 1 >(),
            1 };
      }
      else if constexpr ( ThreadLayout::thread_block_dim == 3 )
      {
         return {
            num_work_items,
            ThreadLayout::template GetBlockDim< 0 >(),
            ThreadLayout::template GetBlockDim< 1 >(),
            ThreadLayout::template GetBlockDim< 2 >() };
      }
      else
      {
         return {
            num_work_items,
            ThreadLayout::GetNumberOfThreads(),
            1,
            1 };
      }
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static GlobalIndex GetThreadIndex()
   {
      static_assert(
         Index < ThreadLayout::thread_block_dim,
         "Thread block index dimension is out of bounds." );

   #ifdef GENDIL_DEVICE_CODE
      if constexpr ( ThreadLayout::thread_block_dim == 1 )
      {
         static_assert( Index == 0 );
         return threadIdx.x;
      }
      else if constexpr ( ThreadLayout::thread_block_dim == 2 )
      {
         if constexpr ( Index == 0 ) { return threadIdx.x; }
         else { return threadIdx.y; }
      }
      else if constexpr ( ThreadLayout::thread_block_dim == 3 )
      {
         if constexpr ( Index == 0 ) { return threadIdx.x; }
         else if constexpr ( Index == 1 ) { return threadIdx.y; }
         else { return threadIdx.z; }
      }
      else
      {
         return ThreadLayout::template GetThreadIndex< Index >( threadIdx.x );
      }
   #else
      return 0;
   #endif
   }

   GENDIL_HOST_DEVICE
   static GlobalIndex GetLinearThreadIndex()
   {
   #ifdef GENDIL_DEVICE_CODE
      if constexpr ( ThreadLayout::thread_block_dim == 0 )
      {
         return 0;
      }
      else if constexpr ( ThreadLayout::thread_block_dim == 1 )
      {
         return threadIdx.x;
      }
      else if constexpr ( ThreadLayout::thread_block_dim == 2 )
      {
         return threadIdx.x +
            ThreadLayout::template GetBlockDim< 0 >() * threadIdx.y;
      }
      else if constexpr ( ThreadLayout::thread_block_dim == 3 )
      {
         return threadIdx.x +
            ThreadLayout::template GetBlockDim< 0 >() *
               ( threadIdx.y +
                 ThreadLayout::template GetBlockDim< 1 >() * threadIdx.z );
      }
      else
      {
         return threadIdx.x;
      }
   #else
      return 0;
   #endif
   }

   GENDIL_HOST_DEVICE
   static GlobalIndex BatchIndex()
   {
      return 0;
   }

   GENDIL_HOST_DEVICE
   static GlobalIndex WorkItemIndex()
   {
   #ifdef GENDIL_DEVICE_CODE
      return blockIdx.x;
   #else
      return 0;
   #endif
   }

   GENDIL_HOST_DEVICE
   static bool IsActive( const GlobalIndex num_work_items )
   {
      return WorkItemIndex() < num_work_items;
   }

   static constexpr size_t SharedMemoryStride(
      const size_t per_work_item_reals )
   {
      return per_work_item_reals;
   }

   static constexpr size_t SharedMemoryBlockSize(
      const size_t per_work_item_reals )
   {
      return per_work_item_reals;
   }

   GENDIL_HOST_DEVICE
   static Real * SharedMemoryForWorkItem(
      Real * shared_data,
      const size_t )
   {
      return shared_data;
   }

   template < typename Lambda >
   static inline void BlockLoop( const GlobalIndex n, Lambda && body )
   {
      if ( n == 0 )
      {
         return;
      }

   #if defined( GENDIL_USE_DEVICE )
      const auto geometry = GetLaunchGeometry( n );
      dim3 gridDim( geometry.grid_x );
      dim3 blockDim( geometry.block_x, geometry.block_y, geometry.block_z );
      size_t sharedMemSize = 0;
      Stream_t stream = 0;

      CheckDeviceLaunchConfiguration( gridDim, blockDim, sharedMemSize );
      GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
         "ThreadFirstKernelConfiguration::BlockLoop: before launch" );
      details::BlockLoopKernel< ThreadFirstKernelConfiguration ><<<
         gridDim,
         blockDim,
         sharedMemSize,
         stream >>>( n, std::forward< Lambda >( body ) );
      GENDIL_CHECK_LAST_DEVICE_LAUNCH(
         "ThreadFirstKernelConfiguration::BlockLoop" );
   #elif defined( GENDIL_USE_MFEM )
      static_assert(
         std::is_invocable_v< Lambda, GlobalIndex >,
         "ThreadFirstKernelConfiguration::BlockLoop expects a one-index body." );

      const auto geometry = GetLaunchGeometry( n );
      if constexpr ( ThreadLayout::thread_block_dim == 3 )
      {
         mfem::forall_3D(
            n,
            geometry.block_x,
            geometry.block_y,
            geometry.block_z,
            std::forward< Lambda >( body ) );
      }
      else
      {
         mfem::forall_2D(
            n,
            geometry.block_x,
            geometry.block_y,
            std::forward< Lambda >( body ) );
      }
   #else
      static_assert(
         dependent_false_v< Lambda >,
         "ThreadFirstKernelConfiguration::BlockLoop() requires a device or "
         "MFEM backend." );
   #endif
   }

   /**
    * @brief Diagnostic-only all-candidate device loop.
    *
    * @details This visits every physical candidate lane. For the legacy
    * ThreadFirst configuration all candidates are active for positive n. The
    * body takes no argument and should query metadata via static accessors.
    * Production code must use BlockLoop.
    */
   template < typename Lambda >
   static inline void CandidateBlockLoop( const GlobalIndex n, Lambda && body )
   {
      if ( n == 0 )
      {
         return;
      }

   #if defined( GENDIL_USE_DEVICE )
      const auto geometry = GetLaunchGeometry( n );
      dim3 gridDim( geometry.grid_x );
      dim3 blockDim( geometry.block_x, geometry.block_y, geometry.block_z );
      size_t sharedMemSize = 0;
      Stream_t stream = 0;

      CheckDeviceLaunchConfiguration( gridDim, blockDim, sharedMemSize );
      GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
         "ThreadFirstKernelConfiguration::CandidateBlockLoop: before launch" );
      details::CandidateBlockLoopKernel< ThreadFirstKernelConfiguration ><<<
         gridDim,
         blockDim,
         sharedMemSize,
         stream >>>( n, std::forward< Lambda >( body ) );
      GENDIL_CHECK_LAST_DEVICE_LAUNCH(
         "ThreadFirstKernelConfiguration::CandidateBlockLoop" );
   #else
      static_assert(
         dependent_false_v< Lambda >,
         "ThreadFirstKernelConfiguration::CandidateBlockLoop() requires a device backend." );
   #endif
   }

   GENDIL_DEVICE
   inline void Sync() const
   {
      GENDIL_SYNC_THREADS();
   }

   GENDIL_DEVICE
   inline void SyncWorkItem() const
   {
      Sync();
   }

   GENDIL_DEVICE
   inline void Synchronize() const
   {
      Sync();
   }

};

} // namespace gendil
