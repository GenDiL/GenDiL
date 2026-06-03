// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"
#include <type_traits>

namespace gendil
{

template < typename KernelConfiguration >
struct is_batched_device_configuration
{
   using config = std::remove_cvref_t< KernelConfiguration >;

   static constexpr bool value = []()
   {
      if constexpr (
         is_device_configuration_v< config > &&
         requires { config::batch_size; } )
      {
         return config::batch_size > 1;
      }
      else
      {
         return false;
      }
   }();
};

template < typename KernelConfiguration >
static constexpr bool is_batched_device_configuration_v =
   is_batched_device_configuration<
      std::remove_cvref_t< KernelConfiguration > >::value;

template < typename KernelConfiguration >
static constexpr bool is_unaudited_operator_configuration_allowed_v =
   !is_batched_device_configuration_v< KernelConfiguration >;

} // namespace gendil

#define GENDIL_REQUIRE_BATCH_SIZE_ONE_FOR_UNAUDITED_OPERATOR(                 \
   KernelConfiguration,                                                       \
   OperatorName )                                                             \
   static_assert(                                                             \
      ::gendil::is_unaudited_operator_configuration_allowed_v<                \
         KernelConfiguration >,                                               \
      OperatorName                                                            \
      " has not been audited for BatchSize > 1 with "                        \
      "DeviceKernelConfiguration. Use BatchSize == 1 or add an explicit "     \
      "audited batched implementation." )
