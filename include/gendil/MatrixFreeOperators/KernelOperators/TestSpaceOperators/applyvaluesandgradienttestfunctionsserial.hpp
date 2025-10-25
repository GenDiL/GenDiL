// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"

namespace gendil {

namespace details
{

template < Integer ActiveDim, typename Pair, typename Op1D, size_t ... Is >
GENDIL_HOST_DEVICE
auto FusedAdjointGradInterpContractionSumReduce( Pair const & u_and_Du, Op1D const & B, std::index_sequence< Is ... > )
{
   auto& u = std::get< 0 >( u_and_Du );
   auto& Du = std::get< 1 >( u_and_Du );

   using UTensor = typename std::decay_t< std::tuple_element_t< 0, Pair > >;
   using DUTensor = typename std::tuple_element_t< 1, Pair >;

   constexpr Integer Rank = get_rank_v< UTensor >;

   SerialRecursiveArray< Real, adjoint_contraction_shape< ActiveDim, Is, UTensor, Op1D >::value ... > Bu;

   constexpr Integer NQ = range_dim_v< Op1D >;

   Loop< adjoint_contraction_shape< ActiveDim, Is, UTensor, Op1D >::value ... >(
      [&] ( auto ... indices_ )
      {
         auto indices = std::make_tuple( indices_ ... );
         const Integer d = std::get< ActiveDim >( indices );

         Real value = Real(0.0);

         auto& q = std::get< ActiveDim >( indices );
         for ( q = 0; q < NQ; ++q )
         {
            const Real b = B.values( q, d );
            const Real g = B.gradients( q, d );

            const Real uq = u( std::get< Is >(indices) ... );
            value += b * uq;

            ConstexprLoop< Rank >(
               [&] (auto l)
               {
                  const Real Duq = Du( std::get< Is >( indices ) ..., l );
                  
                  if constexpr ( l == ActiveDim )
                     value += g * Duq;
                  else
                     value += b * Duq;
               }
            );
         }

         Bu( indices_ ... ) = value;
      }
   );

   return Bu;
}

template < Integer ActiveDim, typename Pair, typename Op1D, size_t ... Is >
GENDIL_HOST_DEVICE
auto FusedAdjointGradInterpContraction( Pair const & u_and_Du, Op1D const & B, std::index_sequence< Is ... > )
{
   auto& u = std::get< 0 >( u_and_Du );
   auto& Du = std::get< 1 >( u_and_Du );

   using UTensor = typename std::decay_t< std::tuple_element_t< 0, Pair > >;
   using DUTensor = typename std::decay_t< std::tuple_element_t< 1, Pair > >;

   constexpr Integer Rank = get_rank_v< UTensor >;

   SerialRecursiveArray< Real, adjoint_contraction_shape< ActiveDim, Is, UTensor, Op1D >::value ... > Bu;
   SerialRecursiveArray< Real, adjoint_contraction_shape< ActiveDim, Is, DUTensor, Op1D >::value ..., Rank > Gu;

   constexpr Integer NQ = range_dim_v< Op1D >;

   Loop< adjoint_contraction_shape< ActiveDim, Is, UTensor, Op1D >::value ... >(
      [&] ( auto ... indices_ )
      {
         auto indices = std::make_tuple( indices_ ... );
         const Integer d = std::get< ActiveDim >( indices );

         Real grad[ Rank ]{};
         Real value = Real(0.0);

         auto& q = std::get< ActiveDim >( indices );
         for ( q = 0; q < NQ; ++q )
         {
            const Real b = B.values( q, d );
            const Real g = B.gradients( q, d );

            const Real uq = u( std::get< Is >(indices) ... );
            value += b * uq;

            ConstexprLoop< Rank >(
               [&] (auto l)
               {
                  const Real Duq = Du( std::get< Is >( indices ) ..., l );
                  
                  if constexpr ( l == ActiveDim )
                     grad[ l ] += g * Duq;
                  else
                     grad[ l ] += b * Duq;
               }
            );
         }

         ConstexprLoop< Rank >( [&] (auto l) { Gu( indices_ ..., l ) = grad[ l ]; });
         Bu( indices_ ... ) = value;
      }
   );

   return std::make_tuple( std::move(Bu), std::move(Gu) );
}

template < Integer ActiveDim, bool SumReduce, typename Pair, typename Op1D >
GENDIL_HOST_DEVICE
inline auto FusedAdjointGradInterpContraction( Pair const & u_and_Du, Op1D const & B )
{
   using UTensor = typename std::decay_t< std::tuple_element_t< 0, Pair > >;

   constexpr Integer Rank = get_rank_v< UTensor >;
   static_assert( ActiveDim <  Rank );

   if constexpr ( SumReduce )
      return FusedAdjointGradInterpContractionSumReduce< ActiveDim >( u_and_Du, B, std::make_index_sequence< Rank >{} );
   else
      return FusedAdjointGradInterpContraction< ActiveDim >( u_and_Du, B, std::make_index_sequence< Rank >{} );
}

template < Integer ActiveDim, typename Pair, typename ElementDofToQuad >
GENDIL_HOST_DEVICE
inline auto ApplyValuesAndGradientTestFunctionsImpl( ElementDofToQuad const & element_quad_data, Pair const & u_and_Du )
{
   using UTensor = typename std::decay_t< std::tuple_element_t< 0, Pair > >;

   constexpr Integer Rank = get_rank_v< UTensor >;
   constexpr bool SumReduce = (ActiveDim == 0);

   auto& B = std::get< ActiveDim >( element_quad_data );

   if constexpr ( ActiveDim+1 == Rank )
      return FusedAdjointGradInterpContraction< ActiveDim, SumReduce >( u_and_Du, B );
   else
      return FusedAdjointGradInterpContraction< ActiveDim, SumReduce >( ApplyValuesAndGradientTestFunctionsImpl< ActiveDim+1 >( element_quad_data, u_and_Du ), B );
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
template < bool Add,
           Integer Rank,
           typename FiniteElementSpace,
           typename IntegrationRule,
           typename ElementDofToQuad >
GENDIL_HOST_DEVICE
void ApplyValuesAndGradientTestFunctions(
   const ElementDofToQuad & element_quad_data,
   const QuadraturePointValues< IntegrationRule > & Duq,
   const QuadraturePointValues< IntegrationRule, Rank > & DGuq,
   ElementDoF< FiniteElementSpace > & dofs_out )
{
   if constexpr ( Add )
      dofs_out.data += details::ApplyValuesAndGradientTestFunctionsImpl< 0 >( element_quad_data, std::tie( Duq, DGuq ) );
   else
      dofs_out.data  = details::ApplyValuesAndGradientTestFunctionsImpl< 0 >( element_quad_data, std::tie( Duq, DGuq ) );
}

template <
   bool Add,
   typename ElementDofToQuad,
   typename ValuesInput,
   typename GradientsInput,
   typename Output >
GENDIL_HOST_DEVICE
void ApplyValuesAndGradientTestFunctions(
   const ElementDofToQuad & element_quad_data,
   const ValuesInput & Duq,
   const GradientsInput & DGuq,
   Output & dofs_out )
{
   if constexpr ( Add )
      dofs_out += details::ApplyValuesAndGradientTestFunctionsImpl< 0 >( element_quad_data, std::tie( Duq, DGuq ) );
   else
      dofs_out  = details::ApplyValuesAndGradientTestFunctionsImpl< 0 >( element_quad_data, std::tie( Duq, DGuq ) );
}

}
