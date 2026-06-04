// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/IndexSequenceHelperFunctions/print.hpp"
#include "gendil/Utilities/KernelContext/KernelConfigurations/helpers.hpp"
#include "gendil/Utilities/KernelContext/threadlayout.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil {

#if defined( GENDIL_USE_CUDA )
static constexpr size_t device_warp_size = 32;
#elif defined( GENDIL_USE_HIP )
// HIP's wavefront size can depend on the AMD GPU architecture or compilation
// mode. Use the common AMD backend default for now and refine when needed.
static constexpr size_t device_warp_size = 64;
#else
static constexpr size_t device_warp_size = 1;
#endif

/**
 * @brief Experimental batched GPU kernel configuration.
 *
 * @details BatchIndex() is the local batch lane inside a thread block. It is
 * not a global element or face id. WorkItemIndex() is the semantic global
 * work-item index: for cell kernels it is the cell/element index, and for
 * global-facet kernels it is the global face index.
 *
 * Physical mapping:
 *  - local logical thread index is decoded from threadIdx.x
 *  - BatchIndex() is threadIdx.y
 *  - WorkItemIndex() is blockIdx.x * BatchSize + BatchIndex()
 *  - block dim is dim3(ThreadLayout::GetNumberOfThreads(), BatchSize)
 *  - grid dim is dim3(ceil_div(num_work_items, BatchSize))
 */
template <
   typename ThreadLayout,
   size_t MaxSharedDimensions,
   size_t BatchSize = 1 >
class DeviceKernelConfiguration
{
public:
   static_assert(
      MaxSharedDimensions >= ThreadLayout::thread_block_dim,
      "DeviceKernelConfiguration requires MaxSharedDimensions to be greater "
      "than or equal to the number of threaded dimensions." );
   static_assert( BatchSize >= 1, "BatchSize must be at least one." );

   using thread_layout_type = ThreadLayout;

   static constexpr bool is_host_configuration = false;
   static constexpr bool is_device_configuration = true;
   static constexpr size_t batch_size = BatchSize;
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

   DeviceKernelConfiguration() = default;

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return ThreadLayout::GetNumberOfThreads(); // FIXME this is per batch, not total threads
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
      return {
         details::CeilDiv( num_work_items, BatchSize ),
         ThreadLayout::GetNumberOfThreads(),
         BatchSize,
         1 };
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static GlobalIndex GetThreadIndex()
   {
      static_assert(
         Index < ThreadLayout::thread_block_dim,
         "Thread block index dimension is out of bounds." );

   #ifdef GENDIL_DEVICE_CODE
      return ThreadLayout::template GetThreadIndex< Index >( threadIdx.x );
   #else
      return 0;
   #endif
   }

   GENDIL_HOST_DEVICE
   static GlobalIndex GetLinearThreadIndex()
   {
   #ifdef GENDIL_DEVICE_CODE
      return threadIdx.x;
   #else
      return 0;
   #endif
   }

   GENDIL_HOST_DEVICE
   static GlobalIndex BatchIndex()
   {
   #ifdef GENDIL_DEVICE_CODE
      return threadIdx.y;
   #else
      return 0;
   #endif
   }

   GENDIL_HOST_DEVICE
   static GlobalIndex WorkItemIndex()
   {
   #ifdef GENDIL_DEVICE_CODE
      return blockIdx.x * BatchSize + BatchIndex();
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
      // Shared-memory arenas are counted in Real slots, so a whole-slot stride
      // keeps each batch-local arena aligned for Real-backed views.
      return per_work_item_reals;
   }

   static constexpr size_t SharedMemoryBlockSize(
      const size_t per_work_item_reals )
   {
      return BatchSize * SharedMemoryStride( per_work_item_reals );
   }

   GENDIL_HOST_DEVICE
   static Real * SharedMemoryForWorkItem(
      Real * shared_data,
      const size_t per_work_item_reals )
   {
      return shared_data + BatchIndex() * SharedMemoryStride(
         per_work_item_reals );
   }

   template < size_t WarpOrWavefrontSize >
   GENDIL_HOST_DEVICE
   static constexpr bool WorkItemThreadsAlignWithWarpOrWavefront()
   {
      static_assert(
         WarpOrWavefrontSize > 0,
         "Warp or wavefront size must be positive." );
      return ThreadLayout::GetNumberOfThreads() % WarpOrWavefrontSize == 0;
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
         "DeviceKernelConfiguration::BlockLoop: before launch" );
      details::BlockLoopKernel< DeviceKernelConfiguration ><<<
         gridDim,
         blockDim,
         sharedMemSize,
         stream >>>( n, std::forward< Lambda >( body ) );
      GENDIL_CHECK_LAST_DEVICE_LAUNCH(
         "DeviceKernelConfiguration::BlockLoop" );
   #else
      static_assert(
         dependent_false_v< Lambda >,
         "DeviceKernelConfiguration::BlockLoop() requires a device backend." );
   #endif
   }

   /**
    * @brief Diagnostic-only all-candidate device loop.
    *
    * @details This visits every physical candidate lane, including inactive
    * lanes in the final partial batch. The body takes no argument and should
    * query metadata via static accessors on DeviceKernelConfiguration. This is
    * not a production execution API; production code must use BlockLoop.
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
         "DeviceKernelConfiguration::CandidateBlockLoop: before launch" );
      details::CandidateBlockLoopKernel< DeviceKernelConfiguration ><<<
         gridDim,
         blockDim,
         sharedMemSize,
         stream >>>( n, std::forward< Lambda >( body ) );
      GENDIL_CHECK_LAST_DEVICE_LAUNCH(
         "DeviceKernelConfiguration::CandidateBlockLoop" );
   #else
      static_assert(
         dependent_false_v< Lambda >,
         "DeviceKernelConfiguration::CandidateBlockLoop() requires a device backend." );
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
