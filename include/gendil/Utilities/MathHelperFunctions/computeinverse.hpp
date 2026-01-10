// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "determinant.hpp"

namespace gendil {

GENDIL_HOST_DEVICE
void ComputeInverse( const Real & A, Real & inv_A )
{
   inv_A = Real(1.0)/A;
}

GENDIL_HOST_DEVICE
void ComputeInverse( const Real & det, const Real (& A)[1][1], Real (& inv_A)[1][1] )
{
   inv_A[0][0] = Real(1.0)/A[0][0];
}

GENDIL_HOST_DEVICE
void ComputeInverse( const Real & det, const Real (& A)[2][2], Real (& inv_A)[2][2] )
{
   Real det_inv = Real(1.0) / det;
   inv_A[0][0] = det_inv * A[1][1];
   inv_A[0][1] = - det_inv * A[0][1];
   inv_A[1][0] = - det_inv * A[1][0];
   inv_A[1][1] = det_inv * A[0][0];
}

GENDIL_HOST_DEVICE
void ComputeInverse( const Real & detA, const Real (& A)[3][3], Real (& inv_A)[3][3] )
{
   Real const detAinv = Real(1.0) / detA;
   inv_A[0][0] = detAinv * ( (A[1][1] * A[2][2]) - (A[1][2] * A[2][1]) );
   inv_A[0][1] = detAinv * ( (A[2][1] * A[0][2]) - (A[0][1] * A[2][2]) );
   inv_A[0][2] = detAinv * ( (A[0][1] * A[1][2]) - (A[1][1] * A[0][2]) );
   inv_A[1][0] = detAinv * ( (A[2][0] * A[1][2]) - (A[1][0] * A[2][2]) );
   inv_A[1][1] = detAinv * ( (A[0][0] * A[2][2]) - (A[0][2] * A[2][0]) );
   inv_A[1][2] = detAinv * ( (A[1][0] * A[0][2]) - (A[0][0] * A[1][2]) );
   inv_A[2][0] = detAinv * ( (A[1][0] * A[2][1]) - (A[2][0] * A[1][1]) );
   inv_A[2][1] = detAinv * ( (A[2][0] * A[0][1]) - (A[0][0] * A[2][1]) );
   inv_A[2][2] = detAinv * ( (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]) );
}

template < Integer N >
GENDIL_HOST_DEVICE
void ComputeInverse( const Real (& A)[N][N], Real (& inv_A)[N][N] )
{
   Real const detA = Determinant( A );
   ComputeInverse( detA, A, inv_A );
}

GENDIL_HOST_DEVICE
void ComputeInverse( const Real & det, const std::array< std::array< Real, 1 >, 1 > & A, std::array< std::array< Real, 1 >, 1 > & inv_A )
{
   inv_A[0][0] = Real(1.0)/A[0][0];
}

GENDIL_HOST_DEVICE
void ComputeInverse( const Real & det, const std::array< std::array< Real, 2 >, 2 > & A, std::array< std::array< Real, 2 >, 2 > & inv_A )
{
   Real det_inv = Real(1.0) / det;
   inv_A[0][0] = det_inv * A[1][1];
   inv_A[0][1] = - det_inv * A[0][1];
   inv_A[1][0] = - det_inv * A[1][0];
   inv_A[1][1] = det_inv * A[0][0];
}

GENDIL_HOST_DEVICE
void ComputeInverse( const Real & det, const std::array< std::array< Real, 3 >, 3 > & A, std::array< std::array< Real, 3 >, 3 > & inv_A )
{
   Real const detAinv = Real(1.0) / det;
   inv_A[0][0] = detAinv * ( (A[1][1] * A[2][2]) - (A[1][2] * A[2][1]) );
   inv_A[0][1] = detAinv * ( (A[2][1] * A[0][2]) - (A[0][1] * A[2][2]) );
   inv_A[0][2] = detAinv * ( (A[0][1] * A[1][2]) - (A[1][1] * A[0][2]) );
   inv_A[1][0] = detAinv * ( (A[2][0] * A[1][2]) - (A[1][0] * A[2][2]) );
   inv_A[1][1] = detAinv * ( (A[0][0] * A[2][2]) - (A[0][2] * A[2][0]) );
   inv_A[1][2] = detAinv * ( (A[1][0] * A[0][2]) - (A[0][0] * A[1][2]) );
   inv_A[2][0] = detAinv * ( (A[1][0] * A[2][1]) - (A[2][0] * A[1][1]) );
   inv_A[2][1] = detAinv * ( (A[2][0] * A[0][1]) - (A[0][0] * A[2][1]) );
   inv_A[2][2] = detAinv * ( (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]) );
}

template < Integer Dim >
GENDIL_HOST_DEVICE
void ComputeInverse(
   const Real & det,
   const std::array< Real, Dim > & A,
   std::array< Real, Dim > & inv_A )
{
   ConstexprLoop< Dim >( [&] ( auto i )
   {
      inv_A[i] = Real(1.0) / A[i];
   });
}

template < Integer N >
GENDIL_HOST_DEVICE
void ComputeInverse(
   const std::array< std::array< Real, N >, N > & A,
   std::array< std::array< Real, N >, N > & inv_A )
{
   Real const detA = Determinant( A );
   ComputeInverse( detA, A, inv_A );
}

template < typename ... MatrixTypes >
GENDIL_HOST_DEVICE
void ComputeInverse( const std::tuple< MatrixTypes... > & A, std::tuple< MatrixTypes... > & inv_A );

template < typename MatrixTuple, size_t... Is >
GENDIL_HOST_DEVICE
void ComputeInverse( const MatrixTuple & A, MatrixTuple & inv_A, std::index_sequence<Is...> )
{
   ( ( ComputeInverse( std::get<Is>( A ), std::get<Is>( inv_A ) ) ),...);
}

template < typename ... MatrixTypes >
GENDIL_HOST_DEVICE
void ComputeInverse( const std::tuple< MatrixTypes... > & A, std::tuple< MatrixTypes... > & inv_A )
{
   return ComputeInverse( A, inv_A, std::make_index_sequence< sizeof...(MatrixTypes) >{} );
}

}
