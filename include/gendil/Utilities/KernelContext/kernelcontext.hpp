// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file kernelcontext.hpp
 * @brief Kernel execution context and per-work-item scratch-size utilities.
 *
 * Shared-memory sizes in this header count Real elements, not bytes.  The
 * sizing traits compute the exact scratch requirement for one semantic work
 * item.  KernelContext then maps that requirement onto the physical execution
 * configuration, including disjoint arenas for batched device work items.
 */

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MemoryManagement/memoryarena.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/subsequence.hpp"
#include "gendil/Utilities/KernelContext/KernelConfigurations/helpers.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

#include <type_traits>
#include <utility>

namespace gendil
{

/**
 * @brief Couple a kernel configuration to a fixed-capacity shared allocator.
 *
 * RequiredSharedMemorySize is the exact scratch capacity for one semantic work
 * item.  A configuration may add a stride between work items and combine
 * several strides into one physical thread-block allocation.  The
 * SharedAllocator always retains the exact per-work-item capacity; the
 * declaration-safe shared_memory_block_size is clamped to one only when the
 * complete block requires no usable shared storage.
 *
 * KernelContext does not own the constructor's @c shared_data pointer.  Every
 * context participating in a physical block must be constructed from the same
 * backing array so the kernel configuration can select the calling work item's
 * disjoint slice.
 *
 * @tparam KernelConfiguration Host or device execution configuration.
 * @tparam RequiredSharedMemorySize Exact per-work-item capacity in Real slots.
 */
template < typename KernelConfiguration, Integer RequiredSharedMemorySize >
class KernelContext : public KernelConfiguration
{
public:
   /// Execution configuration represented by this context.
   using kernel_configuration_type = KernelConfiguration;

   /// Whether the underlying configuration executes on the host.
   static constexpr bool is_host_configuration =
      KernelConfiguration::is_host_configuration;

   /// Whether the underlying configuration executes on a device.
   static constexpr bool is_device_configuration =
      KernelConfiguration::is_device_configuration;

   /// Exact scratch capacity available to one semantic work item.
   static constexpr size_t per_work_item_shared_memory_size =
      RequiredSharedMemorySize;

   /// Distance in Real slots between adjacent work-item arenas.
   static constexpr size_t shared_memory_stride_per_work_item =
      details::shared_memory_stride< KernelConfiguration >::value(
         RequiredSharedMemorySize );

   /**
    * @brief Declaration-safe backing-array extent for one physical block.
    *
    * Nonzero requirements include every batch lane and any configuration
    * padding.  A logically empty allocation reports one so callers can declare
    * a standard-conforming array; the extra slot is not allocator capacity.
    */
   static constexpr size_t shared_memory_block_size =
      []
      {
         constexpr size_t block_size =
            details::shared_memory_block_size< KernelConfiguration >::value(
               RequiredSharedMemorySize );
         return block_size == 0 ? size_t( 1 ) : block_size;
      }();

   /**
    * @brief Non-owning allocator for the calling work item's scratch slice.
    *
    * Its capacity is exactly per_work_item_shared_memory_size.  The allocator
    * returns nullptr when a request exceeds that capacity; the declaration-only
    * fallback slot never increases it.
    */
   MemoryArena< Real, RequiredSharedMemorySize > SharedAllocator;

   /**
    * @brief Construct a context over a physical block's shared backing array.
    *
    * @param shared_data Non-null pointer to at least shared_memory_block_size
    * Real slots, including when the logical requirement is zero.  The execution
    * configuration selects the current work item's slice before initializing
    * SharedAllocator.
    */
   GENDIL_HOST_DEVICE
   KernelContext( Real * shared_data )
      : KernelConfiguration(),
        SharedAllocator(
           details::SharedMemoryForConfiguration< KernelConfiguration >(
              shared_data,
              RequiredSharedMemorySize ) )
   {}
};

/**
 * @brief Number of quadrature entries stored in registers by a kernel type.
 *
 * @tparam KernelContext Context or configuration exposing register dimensions.
 * @tparam IntegrationRule Integration-rule type with a tensor point shape.
 */
template <
   typename KernelContext,
   typename IntegrationRule >
struct register_block_size
{
   static constexpr size_t value = Product( subsequence_t< typename IntegrationRule::points::num_points_tensor, typename KernelContext::RegisterDimensions >{} );
};

/** @brief Register-block size for an explicit tensor shape. */
template <
   typename KernelContext,
   size_t... Sizes >
struct register_block_size< KernelContext, std::index_sequence< Sizes... > >
{
   static constexpr size_t value = Product( subsequence_t< std::index_sequence< Sizes... >, typename KernelContext::template register_dimensions< sizeof...(Sizes) > >{} );
};

/** @brief Convenience value for register_block_size. */
template <
   typename KernelContext,
   typename IntegrationRule >
static constexpr size_t register_block_size_v = register_block_size< KernelContext, IntegrationRule >::value;

/**
 * @brief Number of quadrature entries in one shared tensor work buffer.
 *
 * This is an exact per-work-item count and may be zero when the selected shared
 * tensor shape has a zero extent.
 *
 * @tparam KernelContext Context or configuration exposing shared dimensions.
 * @tparam IntegrationRule Integration-rule type with a tensor point shape.
 */
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

/** @brief Shared tensor-block size for an explicit tensor shape. */
template <
   typename KernelContext,
   size_t... Sizes >
struct shared_block_size< KernelContext, std::index_sequence< Sizes... > >
{
   static constexpr size_t value = Product( subsequence_t< std::index_sequence< Sizes... >, typename KernelContext::template shared_dimensions< sizeof...(Sizes) > >{} );
};

/** @brief Convenience value for shared_block_size. */
template <
   typename KernelContext,
   typename IntegrationRule >
static constexpr size_t shared_block_size_v = shared_block_size< KernelContext, IntegrationRule >::value;

/**
 * @brief Exact shared scratch required by one interpolation work item.
 *
 * Two shared tensor buffers are reserved for the interpolation algorithm.  The
 * result counts Real slots per semantic work item, may be zero, and must be
 * converted through KernelContext::shared_memory_block_size before declaring
 * the physical block's backing array.
 */
template <
   typename KernelContext,
   typename IntegrationRule >
static constexpr size_t required_shared_memory_v = 2 * shared_block_size< KernelContext, IntegrationRule >::value;

// TODO This will need to be generalized when we change the thread layout
/** @brief Map a tensor dimension to the current threaded-dimension index. */
template <
   size_t I,
   typename KernelConfiguration >
struct get_thread_dim_index
{
   static constexpr size_t num_reg_dims = KernelConfiguration::space_dim - KernelConfiguration::Dim;
   static constexpr size_t value = I - num_reg_dims;
};

/** @brief Convenience value for get_thread_dim_index. */
template <
   size_t I,
   typename KernelConfiguration >
static constexpr bool get_thread_dim_index_v = get_thread_dim_index< I, KernelConfiguration >::value;

} // namespace gendil
