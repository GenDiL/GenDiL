// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

namespace detail
{
   template<class T, std::size_t N, std::size_t... I>
   GENDIL_HOST_DEVICE
   constexpr std::array<std::remove_cv_t<T>, N>
      to_array_impl(T (&a)[N], std::index_sequence<I...>)
   {
      return {{a[I]...}};
   }
}
 
template<class T, std::size_t N>
GENDIL_HOST_DEVICE
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&a)[N])
{
   return detail::to_array_impl(a, std::make_index_sequence<N>{});
}

template<class T, T... vals>
GENDIL_HOST_DEVICE
constexpr std::array<std::remove_cv_t<T>, sizeof...(vals)> to_array( std::integer_sequence<T, vals...> seq )
{
   return {{vals...}};
}

}
