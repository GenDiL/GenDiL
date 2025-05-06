// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <tuple>
#include "gendil/Utilities/types.hpp"

namespace gendil
{

/// @brief returns the element at position `Index` of the variadic parameter pack.
template < size_t Index, typename... X >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr decltype(auto) get( X&&... x )
{
   static_assert( Index < sizeof...(X), "Index must be in range of variadic argument list." );
   return std::get< Index >( std::tie( std::forward<X>(x)... ) );
}

template < size_t ... Is, typename... X >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr decltype(auto) get( std::index_sequence<Is...>, std::tuple< X... > && x )
{
   // static_assert( Index < sizeof...(X), "Index must be in range of variadic argument list." );
   return std::tie( std::get< Is >( x )... );
}

template < size_t ... Is, typename... X >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr decltype(auto) get( std::index_sequence<Is...> seq, X&&... x )
{
   // static_assert( Index < sizeof...(X), "Index must be in range of variadic argument list." );
   // return std::tie( std::get< Is >( std::tie( std::forward<X>(x)... ) )... );
   return get( seq, std::tie( std::forward<X>(x)... ) );
}

} // namespace gendil
