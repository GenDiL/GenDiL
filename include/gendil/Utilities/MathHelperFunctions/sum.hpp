// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

/**
 * @brief Performs a sum on an arbitrary number of inputs.
 * 
 * @tparam T The type of the first argument.
 * @tparam Args The types of the remaining arguments.
 * @param t The first argument.
 * @param args The rest of the arguments to be summed.
 * @return constexpr auto The sum of all the arguments.
 */
template < typename T, typename... Args >
GENDIL_HOST_DEVICE
constexpr auto Sum( const T & t, const Args & ... args );

template < typename T >
GENDIL_HOST_DEVICE
constexpr auto Sum( const T & t )
{
    return t;
}

template < typename T1, typename T2 >
GENDIL_HOST_DEVICE
constexpr auto Sum( const T1 & t1, const T2 & t2 )
{
    return t1 + t2;
}

template < typename T, typename... Args >
GENDIL_HOST_DEVICE
constexpr auto Sum( const T & t, const Args & ... args )
{
    return t + Sum( args... );
}

constexpr auto Sum()
{
    return 0;
}

}
