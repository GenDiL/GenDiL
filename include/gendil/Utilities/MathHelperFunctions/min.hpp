// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

template < typename T1, typename T2 >
GENDIL_HOST_DEVICE
constexpr auto Min( const T1 & t1, const T2 & t2 )
{
    return t1 < t2 ? t1 : t2;
}

}
