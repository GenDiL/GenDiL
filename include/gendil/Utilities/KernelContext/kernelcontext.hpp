// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MemoryManagement/memoryarena.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/print.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"

namespace gendil
{

template < typename KernelConfiguration, Integer RequiredSharedMemorySize >
class KernelContext : public KernelConfiguration
{
public:
   MemoryArena< Real, RequiredSharedMemorySize > SharedAllocator; // TODO Replace with generic 

   GENDIL_HOST_DEVICE
   KernelContext( Real * shared_data )
      : KernelConfiguration(),
        SharedAllocator(shared_data)
   {}
};

template <
   typename KernelContext,
   typename IntegrationRule >
struct register_block_size
{
   static constexpr size_t value = Product( subsequence_t< typename IntegrationRule::points::num_points_tensor, typename KernelContext::RegisterDimensions >{} );
};

template <
   typename KernelContext,
   size_t... Sizes >
struct register_block_size< KernelContext, std::index_sequence< Sizes... > >
{
   static constexpr size_t value = Product( subsequence_t< std::index_sequence< Sizes... >, typename KernelContext::template register_dimensions< sizeof...(Sizes) > >{} );
};

template <
   typename KernelContext,
   typename IntegrationRule >
static constexpr size_t register_block_size_v = register_block_size< KernelContext, IntegrationRule >::value;

template <
   typename KernelContext,
   typename IntegrationRule >
struct shared_block_size
{
   static constexpr size_t value =
      Product( subsequence_t<
         typename IntegrationRule::points::num_points_tensor,
         typename KernelContext::template shared_dimensions< IntegrationRule::points::num_points_tensor::size() > >{} );
};

template <
   typename KernelContext,
   size_t... Sizes >
struct shared_block_size< KernelContext, std::index_sequence< Sizes... > >
{
   static constexpr size_t value = Product( subsequence_t< std::index_sequence< Sizes... >, typename KernelContext::template shared_dimensions< sizeof...(Sizes) > >{} );
};

template <
   typename KernelContext,
   typename IntegrationRule >
static constexpr size_t shared_block_size_v = shared_block_size< KernelContext, IntegrationRule >::value;

template <
   typename KernelContext,
   typename IntegrationRule >
static constexpr size_t required_shared_memory_v = 2 * shared_block_size< KernelContext, IntegrationRule >::value;

// TODO This will need to be generalized when we change the thread layout
template <
   size_t I,
   typename KernelConfiguration >
struct is_threaded
{
   static constexpr bool value = (I < KernelConfiguration::space_dim) && (I >= ( KernelConfiguration::space_dim - KernelConfiguration::Dim ) );
};

template <
   size_t I,
   typename KernelConfiguration >
static constexpr bool is_threaded_v = is_threaded< I, KernelConfiguration >::value;

// TODO This will need to be generalized when we change the thread layout
template <
   size_t I,
   typename KernelConfiguration >
struct get_thread_dim_index
{
   static constexpr size_t num_reg_dims = KernelConfiguration::space_dim - KernelConfiguration::Dim;
   static constexpr size_t value = I - num_reg_dims;
};

template <
   size_t I,
   typename KernelConfiguration >
static constexpr bool get_thread_dim_index_v = get_thread_dim_index< I, KernelConfiguration >::value;

} // namespace gendil
