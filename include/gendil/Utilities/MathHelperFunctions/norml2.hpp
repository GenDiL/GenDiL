// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "sqrt.hpp"

namespace gendil{

template < typename Vector >
GENDIL_HOST_DEVICE
Real Norml2( const Vector & vec )
{
   return  Sqrt( Dot( vec, vec ) );
}

template < typename KernelContext, typename Vector >
GENDIL_HOST_DEVICE
Real Norml2( const KernelContext & kernel_conf, const Vector & vec )
{
   return  Sqrt( Dot( kernel_conf, vec, vec ) );
}

template < typename FiniteElementSpace >
GENDIL_HOST_DEVICE
Real Norml2( const ElementDoF< FiniteElementSpace > & u )
{
   Real res = 0.0;
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         Real val = u( indices... );
         res += val*val;
      }
   );

   return Sqrt( res );
}

}