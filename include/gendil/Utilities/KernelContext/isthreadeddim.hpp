// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/KernelContext/kernelcontext.hpp"
#include "gendil/Utilities/KernelContext/threadlayout.hpp"

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

} // namespace gendil
