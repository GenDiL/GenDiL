// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MemoryManagement/memoryarena.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/print.hpp"
#include "gendil/Utilities/KernelContext/KernelConfigurations/helpers.hpp"

namespace gendil
{

template < typename KernelConfiguration, Integer RequiredSharedMemorySize >
class KernelContext : public KernelConfiguration
{
public:
   using kernel_configuration_type = KernelConfiguration;

   static constexpr bool is_host_configuration =
      KernelConfiguration::is_host_configuration;
   static constexpr bool is_device_configuration =
      KernelConfiguration::is_device_configuration;
   static constexpr size_t per_work_item_shared_memory_size =
      RequiredSharedMemorySize;
   static constexpr size_t shared_memory_stride_per_work_item =
      details::shared_memory_stride< KernelConfiguration >::value(
         RequiredSharedMemorySize );
   static constexpr size_t shared_memory_block_size =
      details::shared_memory_block_size< KernelConfiguration >::value(
         RequiredSharedMemorySize );

   MemoryArena< Real, RequiredSharedMemorySize > SharedAllocator; // TODO Replace with generic 

   GENDIL_HOST_DEVICE
   KernelContext( Real * shared_data )
      : KernelConfiguration(),
        SharedAllocator(
           details::SharedMemoryForConfiguration< KernelConfiguration >(
              shared_data,
              RequiredSharedMemorySize ) )
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
   using IR = std::remove_cvref_t<IntegrationRule>;
   static constexpr size_t value =
      Product( subsequence_t<
         typename IR::points::num_points_tensor,
         typename KernelContext::template shared_dimensions< IR::points::num_points_tensor::size() > >{} );
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
