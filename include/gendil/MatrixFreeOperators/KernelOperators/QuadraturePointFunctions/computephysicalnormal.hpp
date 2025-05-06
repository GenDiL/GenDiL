// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"

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

template < typename... MatrixTypes, Integer Dim, Integer Index, int Sign >
GENDIL_HOST_DEVICE
auto ComputePhysicalNormal(
   std::tuple< MatrixTypes... > const & inv_J,
   CanonicalVector< Dim, Index, Sign > )
{
   Real res[ Dim ]{};
   res[ Index ] = Sign;
   ApplyMapping( inv_J, res );
   return to_array( res );
}

template < typename MatrixType, Integer Dim, Integer Index, int Sign >
GENDIL_HOST_DEVICE
auto ComputePhysicalNormal(
   MatrixType const & inv_J,
   CanonicalVector< Dim, Index, Sign > )
{
   Real res[ Dim ]{};
   res[ Index ] = Sign;
   ApplyMapping( inv_J, res );
   return to_array( res );
}

}
