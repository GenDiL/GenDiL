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

/// No-op ResetState for stateless host backends.
///
/// Host backends do not cache sparse-matrix-specific state, so there is
/// nothing to reset. This overload is found for all backends derived from
/// HostMatVecBackend.
inline void ResetState( const HostMatVecBackend & ) noexcept
{ }

/// No-op ResetState for stateless native device backends.
///
/// Native device backends (e.g., NativeDeviceBSRBackend) do not cache
/// sparse-matrix-specific state, so there is nothing to reset. This overload
/// is found for all backends derived from DeviceMatVecBackend, except vendor
/// backends which provide their own more-specific overloads.
inline void ResetState( const DeviceMatVecBackend & ) noexcept
{ }

} // namespace gendil
