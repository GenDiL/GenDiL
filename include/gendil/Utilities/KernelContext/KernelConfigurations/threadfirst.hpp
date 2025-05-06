// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once 

#include "gendil/Utilities/IndexSequenceHelperFunctions/print.hpp"
#include "gendil/Utilities/KernelContext/threadlayout.hpp"

namespace gendil {

/**
 * @brief A kernel configuration where the threaded dimensions are the first dimensions of a problem.
 * 
 * @tparam ThreadLayout The virtual shape of the block of threads, expects the `ThreadBlockLayout` struct.
 * @tparam MaxSharedDimensions The maximum number of dimensions that algorithms can put in shared memory. This is used to determine shared memory usage.
 */
template <
   typename ThreadLayout,
   size_t MaxSharedDimensions >
class ThreadFirstKernelConfiguration : public ThreadLayout
{
public:
   using thread_layout_type = ThreadLayout;

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

   GENDIL_HOST_DEVICE
   ThreadFirstKernelConfiguration()
      : ThreadLayout()
   {
   }
};

}
