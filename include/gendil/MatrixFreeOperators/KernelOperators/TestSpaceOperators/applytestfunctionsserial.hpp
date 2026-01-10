// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionsthreaded.hpp"

namespace gendil {

namespace details
{

template <
   bool Gradient,
   Integer ActiveDim,
   typename InputTensor,
   typename Op1D,
   size_t ... Is >
GENDIL_HOST_DEVICE
auto AdjointInterpContraction( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
{
   SerialRecursiveArray< Real, adjoint_contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... > Bu;

   constexpr Integer NQ = range_dim_v< Op1D >;

   Loop< adjoint_contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... >(
      [&] ( auto ... indices_ )
      {
         auto indices = std::make_tuple( indices_ ... );
         const Integer d = std::get< ActiveDim >( indices );

         Real value = Real(0.0);

         auto& q = std::get< ActiveDim >( indices );
         for ( q = 0; q < NQ; ++q )
         {
            const Real uq = u( std::get< Is >( indices ) ... );

            if constexpr ( Gradient )
               value += B.gradients( q, d ) * uq;
            else
               value += B.values( q, d ) * uq;
         }

         Bu( indices_ ... ) = value;
      }
   );

   return Bu;
}

template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D >
GENDIL_HOST_DEVICE
inline auto AdjointInterpContraction( InputTensor const & u, Op1D const & B )
{
   constexpr Integer Rank = get_rank_v< InputTensor >;
   return AdjointInterpContraction< Gradient, ActiveDim >( u, B, std::make_index_sequence< Rank >{} );
}

template < Integer DiffDim, Integer ActiveDim, typename ElementDofToQuad, typename InputTensor >
GENDIL_HOST_DEVICE
inline auto ApplyTestFunctionsImpl( ElementDofToQuad const & element_quad_data, InputTensor const & u )
{
   constexpr Integer Rank = get_rank_v< InputTensor >;
   static_assert( ActiveDim < Rank );

   auto& B = std::get< ActiveDim >( element_quad_data );
   constexpr bool gradient = ActiveDim == DiffDim;

   if constexpr ( ActiveDim+1 == Rank )
      return AdjointInterpContraction< gradient, ActiveDim >( u, B );
   else
      return AdjointInterpContraction< gradient, ActiveDim >( ApplyTestFunctionsImpl< DiffDim, ActiveDim+1 >( element_quad_data, u ), B );
}

} // namespace details

/**
 * @brief N-dimensional implementation of an operator applying test functions from values at quadrature points.
 * 
 * @tparam Add Specify if values should be accumulated or overwritten.
 * @tparam QuadPointTensor The input container for the values at quadrature points.
 * @tparam DofTensor The output container for the dofs.
 * @tparam ElementDofToQuad A tuple of DofToQuad types.
 * @param element_quad_data The tuple containing data at quadrature point for each dimension.
 * @param Buq The values at quadrature point.
 * @param dofs_out The contribution to the degrees-of-freedom.
 * 
 * @note Assumes tensor finite element with tensor integration rule.
 */
template <
   bool Add,
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename QuadPointTensor,
   typename DofTensor,
   typename ElementDofToQuad >
GENDIL_HOST_DEVICE
void ApplyTestFunctionsSerial(
   ElementDofToQuad const & element_quad_data,
   QuadPointTensor const & Buq,
   DofTensor & dofs_out )
{
   if constexpr ( Add )
      dofs_out += details::ApplyTestFunctionsImpl< DiffDim, 0 >( element_quad_data, Buq );
   else
      dofs_out  = details::ApplyTestFunctionsImpl< DiffDim, 0 >( element_quad_data, Buq );
}

}
