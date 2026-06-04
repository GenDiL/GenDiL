// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil
{

namespace details
{
   template < size_t Index, size_t... BlockDims, size_t... I >
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex LogicalThreadStride( std::index_sequence< I... > )
   {
      using dims = std::index_sequence< BlockDims... >;
      return ( GlobalIndex( 1 ) * ... * GlobalIndex( seq_get_v< I, dims > ) );
   }

   template < size_t Index, size_t... BlockDims >
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex LogicalThreadIndex( GlobalIndex linear_thread_index )
   {
      constexpr GlobalIndex stride =
         LogicalThreadStride< Index, BlockDims... >(
            std::make_index_sequence< Index >{} );
      constexpr GlobalIndex dim = vseq_get_v< Index, BlockDims... >;
      return ( linear_thread_index / stride ) % dim;
   }
} // namespace details

/**
 * @brief Logical per-work-item thread layout.
 *
 * @details ThreadBlockLayout describes the virtual thread structure used by a
 * single semantic work item, such as one element or one global face. It does not
 * know about CUDA/HIP block geometry, grid geometry, blockIdx, threadIdx, or
 * batching. Kernel configuration types own the physical mapping.
 *
 * @tparam BlockDims logical thread dimensions, ordered with dimension 0 varying
 * fastest in the linear thread index.
 */
template < size_t... BlockDims >
class ThreadBlockLayout
{
public:
   static constexpr size_t thread_block_dim = sizeof...( BlockDims );
   using block_dimensions = std::index_sequence< BlockDims... >;

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return ( size_t( 1 ) * ... * BlockDims );
   }

   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      static_assert(
         thread_block_dim == 0,
         "GetBlockDim() without a dimension is only valid for ThreadBlockLayout<>." );
      return 1;
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      static_assert(
         Index < thread_block_dim,
         "Thread block dimension is out of bounds." );
      return vseq_get_v< Index, BlockDims... >;
   }

   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetThreadIndex( GlobalIndex )
   {
      static_assert(
         thread_block_dim == 0,
         "GetThreadIndex(linear_thread_index) without a dimension is only "
         "valid for ThreadBlockLayout<>." );
      return 0;
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetThreadIndex(
      GlobalIndex linear_thread_index )
   {
      static_assert(
         Index < thread_block_dim,
         "Thread block index dimension is out of bounds." );
      return details::LogicalThreadIndex< Index, BlockDims... >(
         linear_thread_index );
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr bool IsThreadedDimension()
   {
      return Index < thread_block_dim;
   }
};

} // namespace gendil
