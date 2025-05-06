// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

namespace detail {

template< typename Index, size_t Is >
struct Cat_t;

template< size_t J, size_t... Is >
struct Cat_t< std::index_sequence< Is... >, J >
{
   using type = std::index_sequence< Is..., J >;
};

template< typename Index, size_t Is >
using Cat = typename Cat_t< Index, Is >::type;

template < typename Lambda, size_t... Is >
GENDIL_HOST_DEVICE
constexpr void ConstexprLoop( Lambda && lambda, std::index_sequence< Is... > )
{
   lambda( std::integral_constant< size_t, Is >{}... );
}

template < typename Lambda, typename Index, size_t... Is >
GENDIL_HOST_DEVICE
constexpr void ConstexprLoop( Lambda && lambda, Index, std::index_sequence< Is... > )
{
   ([&](){
      using new_index = Cat< Index, Is >;
      ConstexprLoop( std::forward< Lambda >( lambda ), new_index{} );
   }(),...);
}

template < int second_dim, int... rest_dims, typename Lambda, typename Index, size_t... Is >
GENDIL_HOST_DEVICE
constexpr void ConstexprLoop( Lambda && lambda, Index, std::index_sequence< Is... > )
{
   ([&](){
      using new_index = Cat< Index, Is >;
      ConstexprLoop< rest_dims... >( std::forward< Lambda >( lambda ), new_index{}, std::make_index_sequence< second_dim >{} );
   }(),...);
}

}

/**
 * @brief Constexpr nested for loop, indices are represented with std::index_sequence< Is... >.
 * 
 * @tparam FirstLoopBound The bound of the first constexpr for loop
 * @tparam LoopBounds The following loop bounds
 * @tparam Lambda The body of the for loop
 * @param lambda The body of the for loop
 */
template < int FirstLoopBound, int... LoopBounds, typename Lambda >
GENDIL_HOST_DEVICE
constexpr void ConstexprLoop( Lambda && lambda )
{
   detail::ConstexprLoop< LoopBounds... >( std::forward< Lambda >( lambda ), std::index_sequence<>{}, std::make_index_sequence< FirstLoopBound >{} );
}

/**
 * @brief Constexpr nested for loop, indices are represented with std::index_sequence< Is... >.
 * This version allows to provide an Index, the Index type should be std::index_sequence< Is... >.
 * 
 * @tparam FirstLoopBound The bound of the first constexpr for loop
 * @tparam LoopBounds The following loop bounds
 * @tparam Lambda The body of the for loop
 * @tparam Index The provided index
 * @param lambda The body of the for loop
 */
template < int FirstLoopBound, int... LoopBounds, typename Lambda, typename Index >
GENDIL_HOST_DEVICE
constexpr void ConstexprLoop( Lambda && lambda, Index )
{
   detail::ConstexprLoop< LoopBounds... >( std::forward< Lambda >( lambda ), Index{}, std::make_index_sequence< FirstLoopBound >{} );
}

}
