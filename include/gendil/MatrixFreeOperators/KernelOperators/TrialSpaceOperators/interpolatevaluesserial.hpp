// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/getrank.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatevaluesthreaded.hpp"

namespace gendil
{

namespace details
{

template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D, size_t ... Is >
GENDIL_HOST_DEVICE
auto InterpContraction( InputTensor const & u, Op1D const & B, std::index_sequence< Is ... > )
{
   SerialRecursiveArray< Real, contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... > Bu;
   
   constexpr Integer ND = domain_dim_v< Op1D >;

   Loop< contraction_shape< ActiveDim, Is, InputTensor, Op1D >::value ... >(
      [&] ( auto ... indices_ )
      {
         // TODO: make_tuple copies all of the indices however all but one index
         // (ActiveDim) needs to be modifiable. The other indices could be tied
         // instead. If the compiler does not recognize this, then this
         // implementation may use twice as many registers as needed.
         auto indices = std::make_tuple( indices_ ... );
         const Integer q = std::get< ActiveDim >( indices );

         Real value = 0.0;
         auto& d = std::get< ActiveDim >( indices );
         for ( d = 0; d < ND; ++d )
         {
            const Real dof = u( std::get< Is >( indices ) ... );
            
            // TODO: Replace with std::function< Real( LocalIndex, LocalIndex ) > ?
            if constexpr ( Gradient )
            {
               const Real g = B.gradients( q, d );
               value += g * dof;
            }
            else
            {
               const Real b = B.values( q, d );
               value += b * dof;
            }
         }

         Bu( indices_ ... ) = value;
      }
   );

   return Bu;
}

template < bool Gradient, Integer ActiveDim, typename InputTensor, typename Op1D >
GENDIL_HOST_DEVICE
inline auto InterpContraction( InputTensor const & u, Op1D const & B )
{
   constexpr Integer Rank = get_rank_v< InputTensor >;
   return InterpContraction< Gradient, ActiveDim >( u, B, std::make_index_sequence< Rank >{} );
}

template < Integer ActiveDim,
           typename ElementDofToQuad,
           typename InputTensor>
GENDIL_HOST_DEVICE
inline auto InterpolateValuesImpl( const ElementDofToQuad & element_quad_data, const InputTensor & u )
{
   constexpr Integer Rank = get_rank_v< InputTensor >;
   static_assert( ActiveDim < Rank );

   auto& B = std::get< ActiveDim >( element_quad_data );

   if constexpr ( ActiveDim+1 == Rank )
      return InterpContraction< false, ActiveDim >( u, B );
   else
      return InterpContraction< false, ActiveDim >( InterpolateValuesImpl< ActiveDim+1 >( element_quad_data, u ), B );
}

} // namespace details

template <
   typename ElementDofToQuad,
   typename DofTensor >
GENDIL_HOST_DEVICE
auto InterpolateValuesSerial(
   const ElementDofToQuad & element_quad_data,
   const DofTensor & u )
{
   return details::InterpolateValuesImpl< 0 >( element_quad_data, u );
}

} // namespace gendil
