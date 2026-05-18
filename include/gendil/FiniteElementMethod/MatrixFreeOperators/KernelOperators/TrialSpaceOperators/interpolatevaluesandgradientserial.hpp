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

template <
   typename ElementQuadToQuad,
   typename InputTensor,
   typename ValuesOutputTensor,
   typename GradientOutputTensor >
GENDIL_HOST_DEVICE
void InterpolateValuesAndGradients(
   const ElementQuadToQuad & element_quad_data,
   const InputTensor & u,
   ValuesOutputTensor & Bu,
   GradientOutputTensor & Gu )
{
   Bu = InterpolateValuesSerial( element_quad_data, u );

   constexpr bool face_interp = is_face_interpolation_v< ProductOperator >;
   if constexpr ( face_interp )
   {
      constexpr Integer Rank = std::tuple_size_v< ElementQuadToQuad >;
      ConstexprLoop< Rank >( [&] ( auto ActiveDim )
      {
         details::InterpolateGradientSumFactSerial< ActiveDim >( u, element_quad_data, Gu );
      } );
   }
   else
   {
      constexpr Integer Rank = std::tuple_size_v< ElementQuadToQuad >;
      ConstexprLoop< Rank >( [&] ( auto ActiveDim )
      {
         auto & B1d = std::get< ActiveDim >( element_quad_data );
         details::GradContractionAtQPoints< ActiveDim >( Bu, Gu, B1d );
      } );
   }
}

}
