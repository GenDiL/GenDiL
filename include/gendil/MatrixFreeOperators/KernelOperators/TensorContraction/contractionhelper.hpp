// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/doftoquad.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/indexsequencehelperfunctions.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/VariadicHelperFunctions/variadichelperfunctions.hpp"

namespace gendil
{

namespace details
{
   template < typename Op1D >
   struct DomainRangeDimensions;

   template < typename ShapeFunctions, typename IntegrationRule >
   struct DomainRangeDimensions< DofToQuad< ShapeFunctions, IntegrationRule > >
   {
      static constexpr Integer RangeDim = DofToQuad< ShapeFunctions, IntegrationRule >::num_quads;
      static constexpr Integer DomainDim = DofToQuad< ShapeFunctions, IntegrationRule >::num_dofs;
   };

   template < typename ShapeFunctions, typename IntegrationRule, typename Face, Integer DimIndex >
   struct DomainRangeDimensions< NonconformingDofToQuad< ShapeFunctions, IntegrationRule, Face, DimIndex > >
   {
      static constexpr Integer RangeDim = NonconformingDofToQuad< ShapeFunctions, IntegrationRule, Face, DimIndex >::num_quads;
      static constexpr Integer DomainDim = NonconformingDofToQuad< ShapeFunctions, IntegrationRule, Face, DimIndex >::num_dofs;
   };

   template < typename Op1D >
   inline constexpr size_t range_dim_v = DomainRangeDimensions< Op1D >::RangeDim;

   template < typename Op1D >
   inline constexpr size_t domain_dim_v = DomainRangeDimensions< Op1D >::DomainDim;

   template < size_t ActiveDim, size_t Dim, typename TensorType, typename Op1D >
   using contraction_shape = typename std::integral_constant< size_t, ( ActiveDim == Dim ) ? range_dim_v< Op1D > : get_tensor_size_v< Dim, TensorType > >;

   template < size_t ActiveDim, size_t Dim, typename TensorType, typename Op1D >
   using adjoint_contraction_shape = typename std::integral_constant< size_t, ( ActiveDim == Dim ) ? domain_dim_v< Op1D > : get_tensor_size_v< Dim, TensorType > >;

   template < typename DofToQuad >
   struct MaxContractionSize;

   template < typename... Op1D >
   struct MaxContractionSize< std::tuple< Op1D... > >
   {
      static constexpr Integer value = ( 1 * ... * std::max( domain_dim_v< Op1D >, range_dim_v< Op1D > ) );
   };

   template < size_t Dim, typename ProductOp, typename ContractedDims>
   struct ContractionShape;

   template < size_t Dim, typename ProductOp, size_t... ContractedDims >
   struct ContractionShape< Dim, ProductOp, std::index_sequence< ContractedDims... > >
   {
      using Op1D = std::tuple_element_t< Dim, ProductOp >;
      static constexpr size_t value = ( (Dim == ContractedDims) || ... || false ) ? range_dim_v< Op1D > : domain_dim_v< Op1D >;
   };

   template < typename ProductOp, typename Is >
   struct ContractionInputShape;

   template < typename ProductOp, size_t... I >
   struct ContractionInputShape< ProductOp, std::index_sequence<I...> >
   {
      using type = std::index_sequence< domain_dim_v< std::tuple_element_t< I, ProductOp > >... >;
   };

   template < typename ProductOp, typename Is >
   struct ContractionOutputShape;

   template < typename ProductOp, size_t... I >
   struct ContractionOutputShape< ProductOp, std::index_sequence<I...> >
   {
      using type = std::index_sequence< range_dim_v< std::tuple_element_t< I, ProductOp > >... >;
   };
} // namespace details

template < typename ProductOp >
using make_contraction_input_shape = typename details::ContractionInputShape< ProductOp, std::make_index_sequence< std::tuple_size_v< ProductOp > > >::type;

template < typename ProductOp >
using make_contraction_output_shape = typename details::ContractionOutputShape< ProductOp, std::make_index_sequence< std::tuple_size_v< ProductOp > > >::type;

template < size_t Dimension, typename Container, typename Sizes, Integer... Strides, typename... Indices >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeDimensionSubView( const FixedStridedView< Container, Sizes, Strides... > & x, Indices... indices )
{
   constexpr Integer Size = seq_get_v< Dimension, Sizes >;
   constexpr Integer Stride = vseq_get_v< Dimension, Strides... >;

   // the offset is the linear index associated to indices[Dimension] = 0.
   get< Dimension >( indices... ) = 0;
   const Integer offset = ( 0 + ... + ( Strides * indices) );

   return MakeView( x.data.data+offset, FixedStridedLayout< std::index_sequence< Size >, Stride>{} );
}

/// @brief Returns a view for the active dimension. If ActiveDim is threaded
/// then places values in shared memory and syncs the threads (the view is to
/// shared memory), otherwise the view is directly into `u`
template < size_t ViewSize, size_t ActiveDim, typename KernelContext, typename TensorType, typename IndexTuple, size_t... Is >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto MakeDimensionView(const KernelContext & ctx, TensorType & u, IndexTuple inds )
{
   if constexpr ( KernelContext::template IsThreaded< ActiveDim >() )
   {
      return [&] ( Integer d ) mutable -> const Real& { return ctx.SharedData[ d ]; };
   }
   else
   {
      return [inds = std::move(inds), &u ] ( Integer d ) mutable -> const Real&
      {
         std::get< ActiveDim >( inds ) = d;
         return std::apply( u, inds );
      };
   }
}

} // namespace gendil