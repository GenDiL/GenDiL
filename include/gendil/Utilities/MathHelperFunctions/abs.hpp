// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once


namespace gendil {

template < typename T >
auto Abs( const T & val )
{
   return std::fabs( val );
}

}
