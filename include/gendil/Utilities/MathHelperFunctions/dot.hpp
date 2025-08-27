// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/Utilities/KernelContext/isserial.hpp"
#include "gendil/Utilities/View/threadedview.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/vector.hpp"

namespace gendil {

template < int size >
GENDIL_HOST_DEVICE
Real Dot( const Real (& u)[ size ], const Real (& v)[ size ] )
{
   Real res = 0.0;
   for (size_t i = 0; i < size; i++)
   {
      res += u[ i ] * v[ i ];
   }
   return res;
}

template < size_t size >
GENDIL_HOST_DEVICE
Real Dot( const Real (& u)[ size ], const std::array< Real, size > & v )
{
   Real res = 0.0;
   for (size_t i = 0; i < size; i++)
   {
      res += u[ i ] * v[ i ];
   }
   return res;
}

template < size_t size >
GENDIL_HOST_DEVICE
Real Dot( const std::array< Real, size > & u, const std::array< Real, size > & v )
{
   Real res = 0.0;
   for (size_t i = 0; i < size; i++)
   {
      res += u[ i ] * v[ i ];
   }
   return res;
}

template < typename FiniteElementSpace >
GENDIL_HOST_DEVICE
Real Dot( const ElementDoF< FiniteElementSpace > & u, const ElementDoF< FiniteElementSpace > & v )
{
   Real res = 0.0;
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         
         res += u( indices... ) * v( indices... );
      }
   );
   return res;
}

template < typename T, Integer ... Dims >
GENDIL_HOST_DEVICE
Real Dot( const SerialRecursiveArray< T, Dims... > & u, const SerialRecursiveArray< T, Dims... > & v )
{
   Real local_res = 0.0;
   UnitLoop< Dims... >(
      [&]( auto... indices )
      {
         local_res += u( indices... ) * v( indices... );
      }
   );
   return local_res;
}

template < typename KernelContext, typename Sizes, typename Container >
GENDIL_HOST_DEVICE
Real Dot( const KernelContext & kernel_conf, const ThreadedView< Sizes, KernelContext, Container > & u, const ThreadedView< Sizes, KernelContext, Container > & v )
{
   #ifdef GENDIL_DEVICE_CODE
   if constexpr ( !is_serial_v< KernelContext > )
   {
      // !FIXME Assumes batch_size = 1
      GENDIL_SHARED Real res; // TODO Use context shared memory
      if( kernel_conf.GetLinearThreadIndex() == 0 ) res = 0.0;
      GENDIL_SYNC_THREADS();
      using tshape = subsequence_t< Sizes, typename KernelContext::template threaded_dimensions< Sizes::size() > >;
      ThreadLoop< tshape >( kernel_conf, [&] ( auto... t )
      {
         Real local_res = Dot( u.data, v.data );
         AtomicAdd( res, local_res );
      });
      GENDIL_SYNC_THREADS();
      return res;
   }
   else
   #endif
   {
      return Dot( u.data, v.data );
   }
}

Real Dot( const Vector & u, const Vector & v )
{
   GENDIL_VERIFY(u.IsHostValid() || u.IsDeviceValid(), "Vector data is not valid on either host or device.");
   GENDIL_VERIFY(v.IsHostValid() || v.IsDeviceValid(), "Vector data is not valid on either host or device.");
   // TODO: Make it device compatible
   const Real* u_ptr( u.ReadHostData() );
   const Real* v_ptr( v.ReadHostData() );
   Real sum = 0.0;
   #pragma omp parallel for reduction(+:sum)
   for (size_t i = 0; i < u.Size(); ++i) {
      sum += u_ptr[i] * v_ptr[i];
   }
   return sum;
}

}
