// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "getoffset.hpp"

namespace gendil {

template < size_t offset, size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real & inv_J, Real (& Gu)[Dim] )
{
   Gu[offset] = inv_J * Gu[offset];
}

template < size_t offset, size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real (& inv_J)[1][1], Real (& Gu)[Dim] )
{
   Gu[offset] = inv_J[0][0] * Gu[offset];
}

template < size_t offset, size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real (& inv_J)[2][2], Real (& Gu)[Dim] )
{
   const Real x = inv_J[0][0] * Gu[offset+0] + inv_J[0][1] * Gu[offset+1];
   const Real y = inv_J[1][0] * Gu[offset+0] + inv_J[1][1] * Gu[offset+1];
   Gu[offset+0] = x;
   Gu[offset+1] = y;
}

template < size_t offset, size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real (& inv_J)[3][3], Real (& Gu)[Dim] )
{
   const Real x = inv_J[0][0] * Gu[offset+0] + inv_J[0][1] * Gu[offset+1] + inv_J[0][2] * Gu[offset+2];
   const Real y = inv_J[1][0] * Gu[offset+0] + inv_J[1][1] * Gu[offset+1] + inv_J[1][2] * Gu[offset+2];
   const Real z = inv_J[2][0] * Gu[offset+0] + inv_J[2][1] * Gu[offset+1] + inv_J[2][2] * Gu[offset+2];
   Gu[offset+0] = x;
   Gu[offset+1] = y;
   Gu[offset+2] = z;
}

template < size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const Real (& inv_J)[Dim][Dim], Real (& Gu)[Dim] )
{
   ApplyMapping<0>( inv_J, Gu );
}

template < size_t offset, size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< std::array< Real, 1 >, 1 > & inv_J, Real (& Gu)[Dim] )
{
   Gu[offset] = inv_J[0][0] * Gu[offset];
}

template < size_t offset, size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< std::array< Real, 2 >, 2 > & inv_J, Real (& Gu)[Dim] )
{
   const Real x = inv_J[0][0] * Gu[offset+0] + inv_J[0][1] * Gu[offset+1];
   const Real y = inv_J[1][0] * Gu[offset+0] + inv_J[1][1] * Gu[offset+1];
   Gu[offset+0] = x;
   Gu[offset+1] = y;
}

template < size_t offset, size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< std::array< Real, 3 >, 3 > & inv_J, Real (& Gu)[Dim] )
{
   const Real x = inv_J[0][0] * Gu[offset+0] + inv_J[0][1] * Gu[offset+1] + inv_J[0][2] * Gu[offset+2];
   const Real y = inv_J[1][0] * Gu[offset+0] + inv_J[1][1] * Gu[offset+1] + inv_J[1][2] * Gu[offset+2];
   const Real z = inv_J[2][0] * Gu[offset+0] + inv_J[2][1] * Gu[offset+1] + inv_J[2][2] * Gu[offset+2];
   Gu[offset+0] = x;
   Gu[offset+1] = y;
   Gu[offset+2] = z;
}

template < size_t Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::array< std::array< Real, Dim >, Dim > & inv_J, Real (& Gu)[Dim] )
{
   ApplyMapping<0>( inv_J, Gu );
}

template < size_t offset = 0, typename... MatrixTypes, Integer Dim >
GENDIL_HOST_DEVICE
void ApplyMapping( const std::tuple< MatrixTypes... > & inv_J,
                   Real (& Gu)[Dim] )
{
   using MatrixTuple = std::tuple< MatrixTypes... >;
   ConstexprLoop< sizeof...(MatrixTypes) >(
      [&] ( auto i )
      {
         constexpr size_t new_offset = offset + GetOffset< MatrixTuple, i >::value;
         ApplyMapping< new_offset >( std::get< i >( inv_J ), Gu );
      }
   ); 
}

}
