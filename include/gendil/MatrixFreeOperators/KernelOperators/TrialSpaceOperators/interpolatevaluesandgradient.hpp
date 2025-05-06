// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolategradient.hpp"

namespace gendil {

/**
 * @brief N-dimensional implementation of an operator interpolating values and gradient values at quadrature points
 * from the given degrees-of-freedom.
 * 
 * @tparam FiniteElementSpace The finite element space.
 * @tparam IntegrationRule The integration rule.
 * @tparam ElementDofToQuad A tuple of DofToQuad types.
 * @param element_quad_data The tuple containing data at quadrature point for each dimension.
 * @param u The input degrees-of-freedom.
 * @param Bu The output field values at quadrature points.
 * @param Gu The ouput field gradient values at quadrature points.
 * 
 * @note Assumes tensor finite element with tensor integration rule.
 */
template < Integer Rank, typename FiniteElementSpace, typename IntegrationRule, typename ElementQuadToQuad >
GENDIL_HOST_DEVICE
void InterpolateValuesAndGradients(
   const ElementQuadToQuad & element_quad_data,
   const ElementDoF< FiniteElementSpace > & u,
   QuadraturePointValues< IntegrationRule > & Bu,
   QuadraturePointValues< IntegrationRule, Rank > & Gu )
{
   Bu.data = details::InterpolateValuesImpl<0>( element_quad_data, u );
   ConstexprLoop< Rank >( [&] ( auto ActiveDim )
   {
      auto & B1d = std::get< ActiveDim >( element_quad_data );
      using Op1D = std::decay_t< decltype( B1d ) >;
      if constexpr ( Op1D::num_quads >= Op1D::num_dofs)
      {
         details::GradContractionAtQPoints< ActiveDim >( Bu.data, Gu, B1d );
      }
      else
      {
         details::InterpolateGradientSumFactSerial< ActiveDim >( u, element_quad_data, Gu );
      }
   } );
}

template < typename KernelContext, typename ProductOperator, typename InputTensor >
GENDIL_HOST_DEVICE
void InterpolateValuesAndGradients(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & u )
{
   constexpr Integer dim = std::tuple_size_v< ProductOperator >;
   using quad_shape = make_contraction_output_shape< ProductOperator >;
   using rdims = typename KernelContext::template register_dimensions< dim >;
   using rshape = subsequence_t< quad_shape, rdims >;
   using shape = cat_t< rshape, std::index_sequence< dim > >;
   // auto Gu = MakeSerialRecursiveArray< Real >( shape{} );
   auto Bu = MakeStaticFIFOView< Real >( rshape{} );
   auto Gu = MakeStaticFIFOView< Real >( shape{} );

   if constexpr ( is_serial_v< KernelContext > )
   {
      // serial interpolation
      InterpolateValuesAndGradients( element_quad_data, u, Bu, Gu );
   }
   else
   {
      // threaded interpolation
      InterpolateValuesAndGradientsThreaded( thread, element_quad_data, u, Gu );
   }

   return std::pair{ Bu, Gu };
}

}
