// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

/**
 * @brief Performs a product on an arbitrary number of inputs.
 * 
 * @tparam T The type of the first argument.
 * @tparam Args The types of the remaining arguments.
 * @param t The first argument.
 * @param args The rest of the arguments to be summed.
 * @return constexpr auto The product of all the arguments.
 */
template < typename T, typename... Args >
GENDIL_HOST_DEVICE
constexpr auto Product( const T & t, const Args & ... args );

template < typename T >
GENDIL_HOST_DEVICE
constexpr auto Product( const T & t )
{
   return t;
}

GENDIL_HOST_DEVICE
constexpr Integer Product( )
{
   return 1;
}

template < typename T1, typename T2 >
GENDIL_HOST_DEVICE
constexpr auto Product( const T1 & t1, const T2 & t2 )
{
   return t1 * t2;
}

template < typename T, typename... Args >
GENDIL_HOST_DEVICE
constexpr auto Product( const T & t, const Args & ... args )
{
   return t * Product( args... );
}

/**
 * @brief Performs the product of all the elements in a tuple.
 * 
 */
template < typename... Args >
GENDIL_HOST_DEVICE
auto Product( const std::tuple<Args ...> & tuple );

template < typename... Args, size_t... Is >
GENDIL_HOST_DEVICE
auto Product( const std::tuple<Args ...> & tuple,
              std::index_sequence<Is...> )
{
   return Product( std::get< Is >( tuple )... );
}

template < typename... Args >
GENDIL_HOST_DEVICE
auto Product( const std::tuple<Args ...> & tuple )
{
   return Product( tuple, std::make_index_sequence< std::tuple_size_v< decltype( tuple ) > >{} );
}

/**
 * @brief Performs the product of all the elements in an array.
 * 
 */
template < typename T, Integer Size >
GENDIL_HOST_DEVICE
auto Product( const std::array<T, Size> & array );

template < typename T, Integer Size, size_t... Is >
GENDIL_HOST_DEVICE
auto Product( const std::array<T, Size> & array,
              std::index_sequence<Is...> )
{
   return Product( array[Is]... );
}

template < typename T, Integer Size >
GENDIL_HOST_DEVICE
auto Product( const std::array<T, Size> & array )
{
   return Product( array, std::make_index_sequence< Size >{} );
}

template < typename T, Integer Size >
GENDIL_HOST_DEVICE
auto Product( const T (& array)[Size] )
{
   return Product( array, std::make_index_sequence< Size >{} );
}

template < size_t... X >
GENDIL_HOST_DEVICE
constexpr auto Product(std::index_sequence<X...>)
{
   return (1 * ... * X);
}

}
