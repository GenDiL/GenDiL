// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/IndexSequenceHelperFunctions/print.hpp"
#include "gendil/Utilities/KernelContext/threadlayout.hpp"
#include <type_traits>

namespace gendil
{

template < size_t BatchSize = 1 >
class HostKernelConfiguration
{
public:
   static_assert( BatchSize >= 1, "BatchSize must be at least one." );

   using thread_layout_type = ThreadBlockLayout<>;

   static constexpr bool is_host_configuration = true;
   static constexpr bool is_device_configuration = false;
   static constexpr size_t batch_size = BatchSize;
   static constexpr size_t thread_block_dim = 0;
   static constexpr size_t shared_block_max_dim = 0;

   template < Integer space_dim >
   using threaded_dimensions = std::index_sequence<>;
   template < Integer space_dim >
   using register_dimensions = std::make_index_sequence< space_dim >;
   template < Integer space_dim >
   using non_shared_register_dimensions = std::make_index_sequence< space_dim >;
   template < Integer space_dim >
   using shared_register_dimensions = std::index_sequence<>;
   template < Integer space_dim >
   using shared_dimensions = std::index_sequence<>;

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return 1;
   }

   GENDIL_HOST_DEVICE
   inline void Synchronize() const
   {
   }

   template < typename Lambda >
   static inline void BlockLoop( const GlobalIndex n, Lambda && body )
   {
      static_assert(
         std::is_invocable_v< Lambda, GlobalIndex >,
         "HostKernelConfiguration::BlockLoop expects a one-index body." );

      #pragma omp parallel for
      for ( GlobalIndex i = 0; i < n; ++i )
      {
         body( i );
      }
   }
};

} // namespace gendil
