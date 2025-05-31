// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/indexsequencehelperfunctions.hpp"
#include "gendil/Utilities/VariadicHelperFunctions/variadichelperfunctions.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tuplehelperfunctions.hpp"

namespace gendil
{

namespace AdjGradHelperFunctions
{
   template <
      typename QuadShape,
      Integer SharedDim,
      typename KernelContext,
      typename InputTensor,
      typename OutputTensor,
      typename Op1D,
      typename SliceIndexTuple >
   GENDIL_HOST_DEVICE
   void AdjointGradContractionShared(
      const KernelContext & thread,
      const InputTensor & sx,
      OutputTensor & GTu,
      const Op1D & B,
      SliceIndexTuple && slice_index )
   {
      constexpr Integer NQ = details::range_dim_v< Op1D >;

      using tshape = subsequence_t< QuadShape, typename KernelContext::template threaded_dimensions< QuadShape::size() > >;
      using rshape = subsequence_t< QuadShape, typename KernelContext::template shared_register_dimensions< QuadShape::size() > >;

      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         const Integer q = get< SharedDim >( t... );

         UnitLoop< rshape >( [&] ( auto... j )
         {
            auto sx_view = MakeDimensionSubView< SharedDim >( sx, t..., j... );

            Real value = 0.0;

            for ( Integer p = 0; p < NQ; ++p )
            {
               const Real g = B.quad_gradients( p, q );
               value += g * sx_view( p );
            }

            auto idx = std::tuple_cat( std::tie( j... ), std::forward<SliceIndexTuple>( slice_index ) );
            std::apply( GTu, idx ) += value;
         });
      });
   }

   template <
      typename QuadShape,
      Integer RegisterDim,
      Integer ActiveDim,
      typename KernelContext,
      typename InputTensor,
      typename OutputTensor,
      typename Op1D >
   GENDIL_HOST_DEVICE
   void AdjointGradContractionRegisters(
      const InputTensor & x,
      OutputTensor & GTu,
      const Op1D & B )
   {
      constexpr Integer NQ = details::range_dim_v< Op1D >;

      using rshape = subsequence_t< QuadShape, typename KernelContext::template register_dimensions< QuadShape::size() > >;

      Loop< rshape >( [&] ( auto... j )
      {
         auto x_view = MakeDimensionSubView< RegisterDim >( x, j..., ActiveDim ); // !FIXME should this be the real ActiveDim?

         const Integer q = get< RegisterDim >( j... );

         Real value = 0.0;

         for ( Integer p = 0; p < NQ; ++p )
         {
            const Real g = B.quad_gradients( p, q );
            value += g * x_view( p );
         }

         GTu( j... ) += value;
      });
   }

   template <
      typename QuadShape,
      Integer ActiveDim,
      typename KernelContext,
      typename RegisterTensor,
      typename SharedTensor,
      typename SliceIndexTuple >
   GENDIL_HOST_DEVICE
   void WriteToShared(
      const KernelContext & thread,
      const RegisterTensor & x,
      SharedTensor & sx,
      SliceIndexTuple && slice_index )
   {
      using rshape_shared = subsequence_t< QuadShape , typename KernelContext::template shared_register_dimensions< QuadShape::size() > >;
      using tshape        = subsequence_t< QuadShape, typename KernelContext::template threaded_dimensions< QuadShape::size() > >;

      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape_shared >( [&] ( auto... j )
         {
            auto idx = std::tuple_cat( std::tie( j... ), std::forward<SliceIndexTuple>(slice_index), std::make_tuple( ActiveDim ) );
            sx( t..., j... ) = std::apply( x, idx );
         });
      });

      thread.Synchronize();
   }
} // namespace AdjGradHelperFunctions

template <
   typename KernelContext,
   typename DofToQuad,
   typename ... ScalardDofTensors,
   size_t ... I >
GENDIL_HOST_DEVICE
auto ApplyGradientTestFunctionsAtQPoints(
   const KernelContext & ctx,
   const DofToQuad & quad_data,
   const std::tuple< ScalardDofTensors ... > & u,
   std::index_sequence< I... > )
{
   return std::make_tuple( ApplyGradientTestFunctionsAtQPoints( ctx, std::get< I >( quad_data), std::get< I>( u ) )... );
}

template <
   typename KernelContext,
   typename DofToQuad,
   typename ... ScalarDofTensors >
GENDIL_HOST_DEVICE
auto ApplyGradientTestFunctionsAtQPoints(
   const KernelContext & ctx,
   const DofToQuad & quad_data,
   const std::tuple< ScalarDofTensors ... > & u )
{
   return ApplyGradientTestFunctionsAtQPoints( ctx, quad_data, u, std::make_index_sequence< sizeof...( ScalarDofTensors ) >{} );
}

template <
   typename KernelContext,
   typename ProductOperator,
   typename InputTensor,
   typename OutputTensor >
GENDIL_HOST_DEVICE
void ApplyGradientTestFunctionsAtQPoints(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & u,
   OutputTensor & GTu )
{
   constexpr Integer Dim = std::tuple_size_v< ProductOperator >;
   constexpr Integer ThreadBlockDim = Min( KernelContext::thread_block_dim, Dim );

   constexpr Integer SharedBlockDim = Min( KernelContext::shared_block_max_dim, Dim );
   static_assert( ThreadBlockDim <= SharedBlockDim, "The dimension of the thread block must not exceed the dimension of the shared memory block." );

   constexpr Integer RegisterBlockDim = Dim - ThreadBlockDim;

   using quad_shape = make_contraction_output_shape< ProductOperator >;

// Threading Strategy
   using ThreadedDimensions = typename KernelContext::template threaded_dimensions< Dim >;
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;

   using NonSharedRegisterDimensions = typename KernelContext::template non_shared_register_dimensions< Dim >;
   using SharedRegisterDimensions = typename KernelContext::template shared_register_dimensions< Dim >;

   using SharedDimensions = typename KernelContext::template shared_dimensions< Dim >;

// Memory use
   using shared_shape = subsequence_t< quad_shape, cat_t< ThreadedDimensions, SharedRegisterDimensions > >;
   constexpr size_t shared_size = Product( shared_shape{} );
   Real * shared_data = thread.SharedAllocator.allocate( shared_size );
   auto sx = MakeFixedFIFOView( shared_data, shared_shape{} );

   using rshape_non_shared = subsequence_t< quad_shape, NonSharedRegisterDimensions >;
   using rshape_shared = subsequence_t< quad_shape, SharedRegisterDimensions >;

// zero output
   using tshape = subsequence_t< quad_shape, ThreadedDimensions >;
   using rshape = subsequence_t< quad_shape, RegisterDimensions >;
   UnitLoop < rshape >( [&] ( auto... j )
   {
      GTu( j... ) = 0.0;
   });

// contraction along threaded dimensions
   UnitLoop< rshape_non_shared >( [&] ( auto... slice_index )
   {
      ConstexprLoop< ThreadBlockDim >( [&] ( auto _c )
      {
         constexpr Integer c = _c;
         constexpr Integer ActiveDim = seq_get_v< c, ThreadedDimensions >;

         AdjGradHelperFunctions::WriteToShared< quad_shape, ActiveDim >( thread, u, sx, std::tie( slice_index... ) );

         using Op1D = std::tuple_element_t< ActiveDim, ProductOperator >;
         auto& B = std::get< ActiveDim >( element_quad_data );

         AdjGradHelperFunctions::AdjointGradContractionShared< quad_shape, c >( thread, sx, GTu, B, std::tie( slice_index... ) );

         if constexpr ( c+1 < ThreadBlockDim )
            thread.Synchronize();
      });
   });

// contraction along non-threaded dimensions

   ConstexprLoop< RegisterBlockDim >( [&] ( auto _c )
   {
      constexpr Integer c = _c;
      constexpr Integer ActiveDim = seq_get_v< c, RegisterDimensions >;

      using Op1D = std::tuple_element_t< ActiveDim, ProductOperator >;
      auto& B = std::get< ActiveDim >( element_quad_data );

      ThreadLoop< tshape >( thread, [&] ( auto... )
      {
         // technically only some threads should perform the contractions (hence
         // the ThreadLoop) but there are no side effects if all the threads
         // perform the contractions since they are entirely in registers.
         // Should we get rid of the ThreadLoop?
         AdjGradHelperFunctions::AdjointGradContractionRegisters< quad_shape, c, ActiveDim, KernelContext >( u, GTu, B );
      });
   });

   thread.SharedAllocator.reset();
}

template <
   typename KernelContext,
   typename ProductOperator,
   typename InputTensor >
GENDIL_HOST_DEVICE
auto ApplyGradientTestFunctionsAtQPoints(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & u )
{
   constexpr Integer dim = std::tuple_size_v< ProductOperator >;
   using quad_shape = make_contraction_output_shape< ProductOperator >;
   using rdims = typename KernelContext::template register_dimensions< dim >;
   using rshape = subsequence_t< quad_shape, rdims >;
   using shape = cat_t< rshape, std::index_sequence< dim > >;
   auto GTu = MakeStaticFIFOView< Real >( rshape{} );

   ApplyGradientTestFunctionsAtQPoints( thread, element_quad_data, u, GTu );
   return GTu;
}

template <
   bool Add,
   typename KernelContext,
   typename ElementDofToQuad,
   typename Input,
   typename Output >
GENDIL_HOST_DEVICE
void ApplyGradientTestFunctionsThreaded(
   const KernelContext & thread,
   const ElementDofToQuad & element_quad_data,
   const Input & DGuq,
   Output & dofs_out )
{
   constexpr bool face_interp = is_face_interpolation_v< ElementDofToQuad >;

   if constexpr ( face_interp )
   {
      constexpr Integer Rank = std::tuple_size_v< ElementDofToQuad >;
      ConstexprLoop<Rank>([&]( auto dim )
      {
         if constexpr ( !Add && dim == 0 )
            dofs_out  = ApplyTestFunctionsThreaded< dim >( thread, element_quad_data, DGuq );
         else
            dofs_out += ApplyTestFunctionsThreaded< dim >( thread, element_quad_data, DGuq );
      } );
   }
   else
   {
      using IntegrationRule = typename ElementDofToQuad::integration_rule;
      auto uq = MakeQuadraturePointValuesContainer( thread, IntegrationRule{} );
      ApplyGradientTestFunctionsAtQPoints( thread, element_quad_data, DGuq, uq );
      if constexpr ( Add )
         dofs_out += ApplyTestFunctionsThreaded( thread, element_quad_data, uq );
      else
         dofs_out  = ApplyTestFunctionsThreaded( thread, element_quad_data, uq );
   }
}

} // namespace gendil
