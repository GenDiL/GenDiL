// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

namespace gendil
{

struct HostMatVecBackend
{ };

struct DeviceMatVecBackend
{ };

template < typename Backend >
inline constexpr bool is_host_matvec_backend_v =
   std::is_base_of_v<
      HostMatVecBackend,
      std::remove_cvref_t< Backend > >;

template < typename Backend >
inline constexpr bool is_device_matvec_backend_v =
   std::is_base_of_v<
      DeviceMatVecBackend,
      std::remove_cvref_t< Backend > >;

} // namespace gendil
