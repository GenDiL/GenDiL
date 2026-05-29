// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/KernelContext/KernelConfigurations/kernelconfigurations.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"
#include <type_traits>

namespace gendil
{

template < typename KernelConfig >
struct is_host_configuration
{
   using config = std::remove_cvref_t< KernelConfig >;
   static constexpr bool value = []()
   {
      if constexpr ( requires { config::is_host_configuration; } )
      {
         return config::is_host_configuration;
      }
      else
      {
         return false;
      }
   }();
};

template < typename KernelConfiguration, Integer SharedBlockSize >
struct is_host_configuration<
   KernelContext< KernelConfiguration, SharedBlockSize > >
{
   static constexpr bool value =
      is_host_configuration< KernelConfiguration >::value;
};

template < typename KernelConfig >
static constexpr bool is_host_configuration_v =
   is_host_configuration< std::remove_cvref_t< KernelConfig > >::value;

template < typename KernelConfig >
struct is_device_configuration
{
   using config = std::remove_cvref_t< KernelConfig >;
   static constexpr bool value = []()
   {
      if constexpr ( requires { config::is_device_configuration; } )
      {
         return config::is_device_configuration;
      }
      else
      {
         return false;
      }
   }();
};

template < typename KernelConfiguration, Integer SharedBlockSize >
struct is_device_configuration<
   KernelContext< KernelConfiguration, SharedBlockSize > >
{
   static constexpr bool value =
      is_device_configuration< KernelConfiguration >::value;
};

template < typename KernelConfig >
static constexpr bool is_device_configuration_v =
   is_device_configuration< std::remove_cvref_t< KernelConfig > >::value;

template < typename KernelConfig >
struct is_serial : is_host_configuration< KernelConfig >
{};

template < typename KernelConfig >
static constexpr bool is_serial_v =
   is_serial< std::remove_cvref_t< KernelConfig > >::value;

} // namespace gendil
