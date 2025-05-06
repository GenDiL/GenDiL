// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "determinant.hpp"
#include "computeinverse.hpp"

namespace gendil {

template < Integer N >
GENDIL_HOST_DEVICE
Real ComputeInverseAndDeterminant( const Real (& A)[N][N], Real (&inv_A)[N][N] )
{
   const Real detA = Determinant( A );
   ComputeInverse( detA, A, inv_A );
   return detA;
}

template < Integer N >
GENDIL_HOST_DEVICE
Real ComputeInverseAndDeterminant(
   const std::array< std::array< Real, N >, N > & A,
   std::array< std::array< Real, N >, N > & inv_A )
{
   const Real detA = Determinant( A );
   ComputeInverse( detA, A, inv_A );
   return detA;
}

template < typename... MatrixTypes >
GENDIL_HOST_DEVICE
Real ComputeInverseAndDeterminant( const std::tuple<MatrixTypes...> & A, std::tuple<MatrixTypes...> &inv_A )
{
   ComputeInverse( A, inv_A );
   return Determinant( A );
}

}
