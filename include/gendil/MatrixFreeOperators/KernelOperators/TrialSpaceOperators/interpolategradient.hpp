// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolategradientserial.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolategradientthreaded.hpp"

namespace gendil {

template <
   typename KernelContext,
   typename DofToQuad,
   typename ... ScalardDofTensors,
   size_t ... I >
GENDIL_HOST_DEVICE
auto InterpolateGradient(
   const KernelContext & ctx,
   const DofToQuad & quad_data,
   const std::tuple< ScalardDofTensors ... > & u,
   std::index_sequence< I... > )
{
   return std::make_tuple( InterpolateGradient( ctx, std::get< I >( quad_data ), std::get< I>( u ) )... );
}

template <
   typename KernelContext,
   typename DofToQuad,
   typename ... ScalarDofTensors >
GENDIL_HOST_DEVICE
auto InterpolateGradient(
   const KernelContext & ctx,
   const DofToQuad & quad_data,
   const std::tuple< ScalarDofTensors ... > & u )
{
   return InterpolateGradient( ctx, quad_data, u, std::make_index_sequence< sizeof...( ScalarDofTensors ) >{} );
}

template < typename KernelContext, typename ProductOperator, typename InputTensor >
GENDIL_HOST_DEVICE
auto InterpolateGradient( const KernelContext & thread, const ProductOperator & element_quad_data, const InputTensor & u )
{
   constexpr Integer dim = std::tuple_size_v< ProductOperator >;
   using quad_shape = make_contraction_output_shape< ProductOperator >;
   using rdims = typename KernelContext::template register_dimensions< dim >;
   using rshape = subsequence_t< quad_shape, rdims >;
   using shape = cat_t< rshape, std::index_sequence< dim > >;
   // auto Gu = MakeSerialRecursiveArray< Real >( shape{} );
   auto Gu = MakeStaticFIFOView< Real >( shape{} );

   if constexpr ( is_serial_v< KernelContext > )
   {
      // serial interpolation
      InterpolateGradientSerial( element_quad_data, u, Gu );
   }
   else
   {
      // threaded interpolation
      InterpolateGradientThreaded( thread, element_quad_data, u, Gu );
   }

   return Gu;
}

}
