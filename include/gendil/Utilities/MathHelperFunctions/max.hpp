// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

template < typename T1, typename T2 >
GENDIL_HOST_DEVICE
constexpr auto Max( const T1 & t1, const T2 & t2 )
{
    return t1 > t2 ? t1 : t2;
}

template < typename T1, typename ... T2 >
GENDIL_HOST_DEVICE
constexpr auto Max( const T1 & t1, const T2 & ... t2 )
{
    const auto max = Max( t2 ... );
    return t1 > max ? t1 : max;
}

}
