// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

template < typename T, size_t N, size_t M, size_t... I, size_t... J >
constexpr std::array<std::remove_cv_t<T>, N+M>
   cat(
      const std::array< T, N > & a,
      const std::array< T, M > & b,
      std::index_sequence<I...>,
      std::index_sequence<J...> )
{
   return {{a[I]...,b[J]...}};
}

template < typename T, size_t N, size_t M >
constexpr auto
   cat(
      const std::array< T, N > & a,
      const std::array< T, M > & b )
{
   return cat( a, b, std::make_index_sequence< N >{}, std::make_index_sequence< M >{} );
}

template < typename T, size_t N, size_t M, size_t... RestSizes >
constexpr auto
   cat(
      const std::array< T, N > & a,
      const std::array< T, M > & b,
      const std::array< T, RestSizes > & ... rest )
{
   return cat( cat( a, b, std::make_index_sequence< N >{}, std::make_index_sequence< M >{} ), rest... );
}

}
