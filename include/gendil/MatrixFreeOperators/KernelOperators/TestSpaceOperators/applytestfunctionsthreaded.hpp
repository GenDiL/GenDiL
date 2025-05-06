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
#include "gendil/Utilities/swap.hpp"

#include "gendil/Utilities/IndexSequenceHelperFunctions/print.hpp"

namespace gendil
{

namespace AdjointInterpHelperFunctions
{
   template <
      bool Gradient,
      Integer ActiveDim,
      typename InputTensor,
      typename OutputTensor,
      typename Op1D,
      size_t... OutputShape >
   GENDIL_HOST_DEVICE
   void AdjointInterpContractionRegisters(
      const InputTensor & x,
      OutputTensor & y,
      const Op1D & B,
      std::index_sequence<OutputShape...> )
   {
      constexpr Integer NQ = details::range_dim_v< Op1D >;

      Loop< OutputShape... >( [&] ( auto... i )
      {
         auto x_view = MakeDimensionSubView< ActiveDim >( x, i... );
         const Integer d = get< ActiveDim >( i... );

         Real BTx = 0;
         for ( LocalIndex q = 0; q < NQ; ++q )
         {
            if constexpr ( Gradient )
               BTx += B.gradients( q, d ) * x_view( q );
            else
               BTx += B.values( q, d ) * x_view( q );
         }

         y( i... ) = BTx;
      });
   }

   template <
      bool Gradient,
      Integer ActiveDim,
      typename ThreadLayout,
      typename InputTensor,
      typename OutputTensor,
      typename Op1D,
      size_t... OutputShape >
   GENDIL_HOST_DEVICE
   auto AdjointInterpContractionShared(
      const ThreadLayout & thread,
      const InputTensor & sx,
      OutputTensor & sy,
      const Op1D & B,
      std::index_sequence<OutputShape...> )
   {
      constexpr Integer NQ = details::range_dim_v< Op1D >;

      ThreadLoop< OutputShape... >( thread, [&] ( auto... j )
      {
         auto sx_view = MakeDimensionSubView< ActiveDim >( sx, j... );
         const Integer d = get< ActiveDim >( j... );

         Real BTx = 0;
         for ( LocalIndex q = 0; q < NQ; ++q )
         {
            if constexpr ( Gradient )
               BTx += B.gradients( q, d ) * sx_view( q );
            else
               BTx += B.values( q, d ) * sx_view( q );
         }

         sy( j... ) = BTx;
      });
   }

   template <
      typename DofShape,
      typename QuadShape,
      typename KernelContext,
      typename RegisterView,
      typename SharedView,
      typename SliceIndexTuple >
   GENDIL_HOST_DEVICE
   void WriteSliceToShared(
      const KernelContext & thread,
      const RegisterView & x,
      SharedView & sx,
      SliceIndexTuple&& slice_index )
   {
      using rshape_shared = subsequence_t< DofShape , typename KernelContext::template shared_register_dimensions< DofShape::size() > >;
      using tshape        = subsequence_t< QuadShape, typename KernelContext::template threaded_dimensions< QuadShape::size() > >;

      ThreadLoop< tshape >(thread, [&] ( auto... t )
      {
         UnitLoop< rshape_shared >( [&] ( auto... j )
         {
            auto idx = std::tuple_cat( std::tie(j...), slice_index );
            sx( t..., j... ) = std::apply( x, idx );
         });
      });

      thread.Synchronize();
   }

   template <
      typename DofShape,
      typename KernelContext,
      typename SharedView,
      typename RegisterView,
      typename SliceIndexTuple >
   GENDIL_HOST_DEVICE
   void ReadSliceFromShared(
      const KernelContext & thread,
      const SharedView & sx,
      RegisterView & x,
      SliceIndexTuple && slice_index )
   {
      using rshape_shared = subsequence_t< DofShape, typename KernelContext::template shared_register_dimensions< DofShape::size() > >;
      using tshape        = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;

      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape_shared >( [&] ( auto... j ) 
         {
            auto idx = std::tuple_cat( std::tie( j... ), std::forward<SliceIndexTuple>( slice_index ) );
            std::apply( x, idx ) = sx( t..., j... );
         });
      });
   }
} // namespace AdjointInterpHelperFunctions

template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename InputTensor >
GENDIL_HOST_DEVICE
auto ApplyTestFunctionsThreaded(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & quad_point_values )
{
   constexpr Integer Dim = std::tuple_size_v< ProductOperator >;
   constexpr Integer ThreadBlockDim = Min( KernelContext::thread_block_dim, Dim );

   constexpr Integer SharedBlockDim = Min( KernelContext::shared_block_max_dim, Dim );
   static_assert(
      ThreadBlockDim <= SharedBlockDim,
      "The dimension of the thread block should not exceed the dimension of the shared memory block." );

   constexpr Integer RegisterBlockDim = Dim - ThreadBlockDim;

// determine threading strategy: which dimensions are threaded?
   using ThreadedDimensions = typename KernelContext::template threaded_dimensions< Dim >;
   using RegisterDimensions = typename KernelContext::template register_dimensions< Dim >;

   using dof_shape  = make_contraction_input_shape< ProductOperator >;
   using quad_shape = make_contraction_output_shape< ProductOperator >;

// Since the dimension of the shared block is assumed greater than (or equal to)
// the dimension of the thread block, when we copy values into shared memory,
// each thread copies SharedBlockDim-ThreadBlockDim register dimensions into
// shared. The choice of which dimensions is arbitrary, so we proceed with the
// tail dimensions because it leads to a natural ordering of indices.

   using NonSharedRegisterDimensions = typename KernelContext::template non_shared_register_dimensions< Dim >;
   using SharedRegisterDimensions    = typename KernelContext::template shared_register_dimensions< Dim >;

   using SharedDimensions = typename KernelContext::template shared_dimensions< Dim >;
   
// Memory layout
   // assumes quad_point_values was allocated with rg
   // Real * register_buffer1 = quad_point_values.data.data;
   // Real * register_buffer2 = thread.RegisterAllocator.allocate();
   // constexpr Integer buffer_size = KernelContext::RegisterBlockSize;
   using max_shape = max_sequence_t< quad_shape, dof_shape >;
   constexpr Integer register_buffer_size = register_block_size_v< KernelContext, max_shape >;
   Real buffer1 [ register_buffer_size ];
   Real buffer2 [ register_buffer_size ];
   Real * register_buffer1 = buffer1;
   Real * register_buffer2 = buffer2;
   // TODO load input
   using reg_input_shape = subsequence_t< quad_shape, RegisterDimensions >;
   auto tmp_in = MakeFixedFIFOView( register_buffer1, reg_input_shape{} );
   if constexpr ( DiffDim < Dim )
   {
      UnitLoop< reg_input_shape >( [&]( auto ... indices )
      {
         tmp_in( indices... ) = quad_point_values( indices..., DiffDim );
      });
   }
   else
   {
      UnitLoop< reg_input_shape >( [&]( auto ... indices )
      {
         tmp_in( indices... ) = quad_point_values( indices... );
      });
   }

   constexpr size_t shared_buffer_size = shared_block_size_v< KernelContext, max_shape >;
   Real * shared_buffer1 = thread.SharedAllocator.allocate( shared_buffer_size );
   Real * shared_buffer2 = thread.SharedAllocator.allocate( shared_buffer_size );

// contraction along register dimensions
   using reg_input_shape  = subsequence_t< quad_shape, RegisterDimensions >;
   using reg_output_shape = subsequence_t< dof_shape , RegisterDimensions >;

   using tshape = subsequence_t< quad_shape, ThreadedDimensions >;

   ConstexprLoop< RegisterBlockDim >( [&] ( auto _c )
   {
      constexpr Integer c = _c; // NVCC fails to correctly capture arguments to three or more nested lambda expression, so this line is necessary
      constexpr Integer ActiveDim = seq_get_v<c, RegisterDimensions >;

      using Op1D = std::tuple_element_t< ActiveDim, ProductOperator >;
      auto& B = std::get< ActiveDim >( element_quad_data );

      using contracted_dims = std::make_index_sequence< c >;
      using input_shape = replace_subsequence_t< reg_input_shape, reg_output_shape, contracted_dims >;
      auto x = MakeFixedFIFOView( register_buffer1, input_shape{} );

      using output_contracted_dims = std::make_index_sequence< c+1 >;
      using output_shape = replace_subsequence_t< reg_input_shape, reg_output_shape, output_contracted_dims >;
      auto y = MakeFixedFIFOView( register_buffer2, output_shape{} );

      ThreadLoop< tshape >( thread, [&] ( auto... t ) 
      {
         // technically only some threads should perform the contractions (hence
         // the ThreadLoop) but there are no side effects if all the threads
         // perform the contractions since they are entirely in registers.
         // Should we get rid of the ThreadLoop?
         constexpr bool gradient = ActiveDim == DiffDim;
         AdjointInterpHelperFunctions::AdjointInterpContractionRegisters< gradient, c >( x, y, B, output_shape{} );
      });

      Swap( register_buffer1, register_buffer2 );
   });

// contraction along non-register dimensions
   using shared_input_shape = cat_t< subsequence_t< quad_shape, ThreadedDimensions >, subsequence_t< dof_shape, SharedRegisterDimensions > >;
   using shared_output_shape = subsequence_t< dof_shape, SharedDimensions >;

   using rshape_non_shared = subsequence_t< dof_shape, NonSharedRegisterDimensions >;
   using rshape_shared = subsequence_t< dof_shape, SharedRegisterDimensions >;

   using rshape = subsequence_t< dof_shape, RegisterDimensions >;
   auto x = MakeFixedFIFOView( register_buffer1, rshape{} );
   auto y = MakeFixedFIFOView( register_buffer2, rshape{} );

   UnitLoop< rshape_non_shared >( [&] ( auto... slice_index )
   {
      ConstexprLoop< ThreadBlockDim >( [&] ( auto _c )
      {
         constexpr Integer c = _c; // NVCC fails to correctly capture arguments to three or more nested lambda expression, so this line is necessary
         constexpr Integer ActiveDim = seq_get_v< c, ThreadedDimensions >;

         using Op1D = std::tuple_element_t< ActiveDim, ProductOperator >;
         auto& B = std::get< ActiveDim >( element_quad_data );

         using contracted_dims = std::make_index_sequence< c >;
         using input_shape = replace_subsequence_t< shared_input_shape, shared_output_shape, contracted_dims >;
         auto sx = MakeFixedFIFOView( shared_buffer1, input_shape{} );

         if constexpr ( c == 0 )
         {
            AdjointInterpHelperFunctions::WriteSliceToShared< dof_shape, quad_shape>( thread, x, sx, std::tie( slice_index... ) );
         }

         using output_contracted_dims = std::make_index_sequence< c+1 >;
         using output_shape = replace_subsequence_t< shared_input_shape, shared_output_shape, output_contracted_dims >;
         auto sy = MakeFixedFIFOView( shared_buffer2, output_shape{} );

         constexpr bool gradient = ActiveDim == DiffDim;
         AdjointInterpHelperFunctions::AdjointInterpContractionShared< gradient, c >( thread, sx, sy, B, output_shape{} );

         if constexpr ( c+1 < ThreadBlockDim )
         {
            thread.Synchronize();
            Swap( shared_buffer1, shared_buffer2 );
         }
         else // write slice to registers
         {
            AdjointInterpHelperFunctions::ReadSliceFromShared< dof_shape >( thread, sy, y, std::tie( slice_index... ) );
         }
      });
   });

   thread.SharedAllocator.reset();

   // return y;

   // TODO remove
   auto result = MakeSerialRecursiveArray< Real >( rshape{} );
   UnitLoop< reg_output_shape >( [&]( auto ... indices )
   {
      result( indices... ) = y( indices... );
   });
   return result;
}

} // namespace gendil
