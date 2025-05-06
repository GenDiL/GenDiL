// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

GENDIL_HOST_DEVICE
Real Determinant( const Real & A )
{
    return A;
}

GENDIL_HOST_DEVICE
Real Determinant( const Real (& A)[1][1] )
{
    return A[0][0];
}

GENDIL_HOST_DEVICE
Real Determinant( const Real (& A)[2][2] )
{
    return A[0][0]*A[1][1] - A[1][0]*A[0][1];
}

GENDIL_HOST_DEVICE
Real Determinant( const Real (& A)[3][3] )
{
    return A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2])
           - A[1][0] * (A[0][1] * A[2][2] - A[2][1] * A[0][2])
           + A[2][0] * (A[0][1] * A[1][2] - A[1][1] * A[0][2]);
}

GENDIL_HOST_DEVICE
Real Determinant( const std::array< std::array< Real, 1 >, 1 > & A )
{
    return A[0][0];
}

GENDIL_HOST_DEVICE
Real Determinant( const std::array< std::array< Real, 2 >, 2 > & A )
{
    return A[0][0]*A[1][1] - A[1][0]*A[0][1];
}

GENDIL_HOST_DEVICE
Real Determinant( const std::array< std::array< Real, 3 >, 3 > & A )
{
    return A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2])
           - A[1][0] * (A[0][1] * A[2][2] - A[2][1] * A[0][2])
           + A[2][0] * (A[0][1] * A[1][2] - A[1][1] * A[0][2]);
}

template < typename ... MatrixTypes >
GENDIL_HOST_DEVICE
Real Determinant( const std::tuple< MatrixTypes... > & A);

template < typename MatrixTuple, size_t... Is >
GENDIL_HOST_DEVICE
Real Determinant( const MatrixTuple & A, std::index_sequence<Is...> )
{
    return Product( Determinant( std::get<Is>( A ) )... );
}

template < typename ... MatrixTypes >
GENDIL_HOST_DEVICE
Real Determinant( const std::tuple< MatrixTypes... > & A)
{
    return Determinant( A, std::make_index_sequence< sizeof...(MatrixTypes) >{} );
}

}
