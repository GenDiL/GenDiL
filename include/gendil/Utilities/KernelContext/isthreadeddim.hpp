// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/KernelContext/kernelcontext.hpp"
#include "gendil/Utilities/KernelContext/threadlayout.hpp"

#include <type_traits>

namespace gendil
{

template < size_t I, typename KernelConfig >
struct is_threaded_dim
{
   static constexpr bool value =
      KernelConfig::thread_layout_type::template IsThreadedDimension< I >();
};

template < size_t I, size_t... Dims >
struct is_threaded_dim< I, ThreadBlockLayout< Dims... > >
{
   static constexpr bool value =
      ThreadBlockLayout< Dims... >::template IsThreadedDimension< I >();
};

template < size_t I, typename KernelConfiguration, Integer SharedBlockSize >
struct is_threaded_dim<
   I,
   KernelContext< KernelConfiguration, SharedBlockSize > >
{
   static constexpr bool value =
      is_threaded_dim< I, KernelConfiguration >::value;
};

template < size_t I, typename KernelConfig >
static constexpr bool is_threaded_dim_v =
   is_threaded_dim< I, KernelConfig >::value;

/**
 * @brief True when a kernel context/configuration has a declared logical
 * per-work-item thread layout.
 *
 * @details This is not a host/device placement trait. It answers whether a
 * helper should use a threaded/shared-memory implementation. Host
 * configurations and ThreadBlockLayout<> device configurations return false.
 * ThreadBlockLayout<1> returns true because it declares one logical threaded
 * dimension, even though its extent is one.
 */
template < typename KernelContext >
struct is_threaded
{
   using context = std::remove_cvref_t< KernelContext >;
   static constexpr bool value =
      []()
      {
         if constexpr (
            requires
            {
               context::is_host_configuration;
               context::thread_block_dim;
            } )
         {
            return !context::is_host_configuration &&
                   context::thread_block_dim > 0;
         }
         else
         {
            return false;
         }
      }();
};

template < typename KernelContext >
static constexpr bool is_threaded_v =
   is_threaded< KernelContext >::value;

} // namespace gendil
