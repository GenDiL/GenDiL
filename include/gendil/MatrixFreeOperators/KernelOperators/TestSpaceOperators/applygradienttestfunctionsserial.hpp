// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctionsthreaded.hpp"

namespace gendil
{

namespace details
{

template < Integer ActiveDim, bool SumReduce, typename InputTensor, typename Op1D, size_t ... Is >
GENDIL_HOST_DEVICE
auto AdjointGradientContraction( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
{
   constexpr Integer Rank = InputTensor::Dim - 1;

   auto Bu = [&]() constexpr {
      if constexpr ( SumReduce )
         return SerialRecursiveArray< Real, adjoint_contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... >{};
      else
         return SerialRecursiveArray< Real, adjoint_contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ..., Rank >{};
   }();
   
   constexpr Integer NQ = range_dim_v< Op1D >;

   Loop< adjoint_contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... >(
      [&] ( auto ... indices_ )
      {
         auto indices = std::make_tuple( indices_ ... );
         const Integer d = std::get< ActiveDim >( indices );

         Real grad[ Rank ]{};
         auto& q = std::get< ActiveDim >( indices );
         
         for ( q = 0; q < NQ; ++q )
         {
            const Real b = B.values( q, d );
            const Real g = B.gradients( q, d );

            ConstexprLoop< Rank >(
               [&] (auto l)
               {
                  const Real dof = u( std::get< Is >( indices ) ..., l );
                  
                  if constexpr ( l == ActiveDim )
                     grad[ l ] += g * dof;
                  else
                     grad[ l ] += b * dof;
               }
            );
         }

         if constexpr ( SumReduce )
         {
            Real sum = 0.0;
            ConstexprLoop< Rank >( [&] (auto l) { sum += grad[ l ]; } );
            Bu( indices_ ... ) = sum;
         }
         else
         {
            ConstexprLoop< Rank >( [&] (auto l) { Bu( indices_ ..., l ) = grad[l]; });
         }
      }
   );

   return Bu;
}

template < Integer ActiveDim, bool SumReduce, typename InputTensor, typename Op1D >
GENDIL_HOST_DEVICE
inline auto AdjointGradientContraction( InputTensor const & u, Op1D const & B )
{
   constexpr Integer Rank = InputTensor::Dim - 1;
   static_assert( Rank > 1 );
   static_assert( ActiveDim <  Rank );

   return AdjointGradientContraction< ActiveDim, SumReduce >( u, B, std::make_index_sequence< Rank >{} );
}

template < Integer ActiveDim, typename InputTensor, typename ElementDofToQuad >
GENDIL_HOST_DEVICE
inline auto ApplyGradientTestFunctionsImpl( const ElementDofToQuad & element_quad_data, const InputTensor & u )
{
   constexpr Integer Rank = InputTensor::Dim - 1;
   constexpr bool SumReduce = (ActiveDim == 0);

   auto& B = std::get< ActiveDim >( element_quad_data );

   if constexpr ( ActiveDim+1 == Rank )
      return AdjointGradientContraction< ActiveDim, SumReduce >( u, B );
   else
      return AdjointGradientContraction< ActiveDim, SumReduce >( ApplyGradientTestFunctionsImpl< ActiveDim+1 >( element_quad_data, u ), B );
}

} // namespace details

/**
 * @brief N-dimensional implementation of an operator applying test functions and gradient of the test functions 
 * from values at quadrature points.
 * 
 * @tparam Add Specify if values should be accumulated or overwritten.
 * @tparam FiniteElementSpace The finite element space.
 * @tparam IntegrationRule The integration rule.
 * @tparam ElementDofToQuad A tuple of DofToQuad types.
 * @param element_quad_data The tuple containing data at quadrature point for each dimension.
 * @param Duq The values at quadrature point.
 * @param dofs_out The contribution to the degrees-of-freed4.
 * 
 * @note Assumes tensor finite element with tensor integration rule.
 */
template <
   bool Add,
   Integer Rank,
   typename FiniteElementSpace,
   typename IntegrationRule,
   typename ElementDofToQuad >
GENDIL_HOST_DEVICE
void ApplyGradientTestFunctions(
   const ElementDofToQuad & element_quad_data,
   const QuadraturePointValues< IntegrationRule, Rank > & DGuq,
   ElementDoF< FiniteElementSpace > & dofs_out )
{
   static_assert( QuadraturePointValues< IntegrationRule, Rank >::Dim > 1 );

   if constexpr ( Add )
      dofs_out.data += details::ApplyGradientTestFunctionsImpl< 0 >( element_quad_data, DGuq );
   else
      dofs_out.data  = details::ApplyGradientTestFunctionsImpl< 0 >( element_quad_data, DGuq );
}

template <
   bool Add,
   typename ElementDofToQuad,
   typename Input,
   typename Output >
GENDIL_HOST_DEVICE
void ApplyGradientTestFunctions(
   const ElementDofToQuad & element_quad_data,
   const Input & DGuq,
   Output & dofs_out )
{
   if constexpr ( Add )
      dofs_out += details::ApplyGradientTestFunctionsImpl< 0 >( element_quad_data, DGuq );
   else
      dofs_out  = details::ApplyGradientTestFunctionsImpl< 0 >( element_quad_data, DGuq );
}

}
