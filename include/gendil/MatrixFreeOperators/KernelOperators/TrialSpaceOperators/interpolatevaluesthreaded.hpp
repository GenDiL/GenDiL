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

namespace gendil
{

namespace InterpHelperFunctions
{
   template <
      bool Gradient,
      Integer ActiveDim,
      typename ThreadLayout,
      typename InputTensor,
      typename OutputTensor,
      typename Op1D,
      size_t... OutputShape >
   GENDIL_HOST_DEVICE
   void InterpContractionShared(
      const ThreadLayout & thread,
      const InputTensor & sx,
      OutputTensor & sy,
      const Op1D & B,
      std::index_sequence< OutputShape... > )
   {
      constexpr Integer ND = details::domain_dim_v< Op1D >;

      // TODO Using `if` instead of `for` might lead to better performance.
      // !FIXME ThreadLoop does not do the right thing here.
      ThreadLoop< OutputShape... >( thread, [&] ( auto... j )
      {
         auto sx_view = MakeDimensionSubView< ActiveDim >( sx, j... );
         const Integer q = get< ActiveDim >( j... );

         Real value = 0.0;

         for ( LocalIndex d = 0; d < ND; ++d )
         {
            if constexpr ( Gradient )
               value += B.gradients( q, d ) * sx_view( d );
            else
               value += B.values( q, d ) * sx_view( d );
         }
         sy( j... ) = value;
      });
   }

   template <
      bool Gradient,
      Integer ActiveDim,
      typename InputTensor,
      typename OutputTensor,
      typename Op1D,
      size_t... OutputShape >
   GENDIL_HOST_DEVICE
   void InterpContractionRegisters(
      const InputTensor & x,
      OutputTensor & y,
      const Op1D & B,
      std::index_sequence< OutputShape... > )
   {
      constexpr Integer ND = details::domain_dim_v< Op1D >;

      // TODO Use ConstexprLoop
      Loop< OutputShape... >( [&] ( auto... j )
      {
         auto x_view = MakeDimensionSubView< ActiveDim >( x, j... );
         const Integer q = get< ActiveDim >( j... );

         Real value = 0.0;

         // TODO Unroll with constexpr loop?
         for ( LocalIndex d = 0; d < ND; ++d )
         {
            if constexpr ( Gradient )
               value += B.gradients( q, d ) * x_view( d );
            else
               value += B.values( q, d ) * x_view( d );
         }
         y( j... ) = value;
      });
   }

   template <
      typename DofShape,
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
      using rshape_shared = subsequence_t< DofShape, typename KernelContext::template shared_register_dimensions< DofShape::size() > >;
      using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;

      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape_shared >( [&] ( auto... j )
         {
            auto idx = std::tuple_cat( std::tie( j... ), std::forward<SliceIndexTuple>( slice_index ) );
            sx( t..., j... ) = std::apply( x, idx );
         });
      });

      thread.Synchronize();
   }

   template <
      typename DofShape,
      typename QuadShape,
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
      using rshape_shared = subsequence_t< DofShape , typename KernelContext::template shared_register_dimensions< DofShape::size() > >;
      using tshape        = subsequence_t< QuadShape, typename KernelContext::template threaded_dimensions< QuadShape::size() > >;

      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape_shared >( [&] ( auto... j )
         {
            auto idx = std::tuple_cat( std::tie( j... ), std::forward<SliceIndexTuple>( slice_index ) );
            std::apply( x, idx ) = sx( t..., j... );
         });
      });
   }
} // namespace InterpHelperFunctions

template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename InputTensor >
GENDIL_HOST_DEVICE
auto InterpolateValuesThreaded(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & element_dofs )
{
   constexpr Integer Dim = std::tuple_size_v< ProductOperator >;
   constexpr Integer ThreadBlockDim = Min( KernelContext::thread_block_dim, Dim );

   constexpr Integer SharedBlockDim = Min( KernelContext::shared_block_max_dim, Dim );
   static_assert(
      ThreadBlockDim <= SharedBlockDim,
      "The dimension of the thread block should not exceed the dimension of the shared memory block." );

   constexpr Integer RegisterBlockDim = Dim - ThreadBlockDim;

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
   using SharedRegisterDimensions = typename KernelContext::template shared_register_dimensions< Dim >;

   using SharedDimensions = typename KernelContext::template shared_dimensions< Dim >;

// Memory layout
   using max_shape = max_sequence_t< quad_shape, dof_shape >;
   constexpr Integer register_buffer_size = register_block_size_v< KernelContext, max_shape >;
   Real buffer1 [ register_buffer_size ];
   Real buffer2 [ register_buffer_size ];
   Real * register_buffer1 = buffer1;
   Real * register_buffer2 = buffer2;

   constexpr size_t shared_buffer_size = shared_block_size_v< KernelContext, max_shape >;
   Real * shared_buffer1 = thread.SharedAllocator.allocate( shared_buffer_size );
   Real * shared_buffer2 = thread.SharedAllocator.allocate( shared_buffer_size );

// contraction along non-register dimensions
   using tshape = subsequence_t<quad_shape, ThreadedDimensions>;
   using shared_input_shape = subsequence_t< dof_shape, SharedDimensions >;
   using shared_output_shape = cat_t< subsequence_t< quad_shape, ThreadedDimensions >, subsequence_t< dof_shape, SharedRegisterDimensions > >;

   using rshape_non_shared = subsequence_t< dof_shape, NonSharedRegisterDimensions >;
   using rshape_shared = subsequence_t< dof_shape, SharedRegisterDimensions >;

   auto y = MakeFixedFIFOView( register_buffer2, subsequence_t< dof_shape, RegisterDimensions >{} );

   UnitLoop< rshape_non_shared >( [&] ( auto... slice_index )
   {
      ConstexprLoop< ThreadBlockDim >( [&] ( auto _c )
      {
         constexpr Integer c = _c; // NVCC fails to correctly capture arguments to nested lambda expression, so this line is necessary
         constexpr Integer ActiveDim = seq_get_v< c, ThreadedDimensions >;

         using Op1D = std::tuple_element_t< ActiveDim, ProductOperator >;
         auto& B = std::get< ActiveDim >( element_quad_data );

         using contracted_dims = std::make_index_sequence< c >;
         using input_shape = replace_subsequence_t< shared_input_shape, shared_output_shape, contracted_dims >;
         auto sx = MakeFixedFIFOView( shared_buffer1, input_shape{} );

         if constexpr ( c == 0 )
         {
            InterpHelperFunctions::WriteSliceToShared< dof_shape >( thread, element_dofs, sx, std::tie( slice_index... ) );
         }

         using output_contracted_dims = std::make_index_sequence< c+1 >;
         using output_shape = replace_subsequence_t< shared_input_shape, shared_output_shape, output_contracted_dims >;
         auto sy = MakeFixedFIFOView( shared_buffer2, output_shape{} );

         constexpr bool gradient = ActiveDim == DiffDim;
         InterpHelperFunctions::InterpContractionShared< gradient, c >( thread, sx, sy, B, output_shape{} );

         if constexpr ( c+1 < ThreadBlockDim )
         {
            thread.Synchronize();
            Swap( shared_buffer1, shared_buffer2 );
         }
         else // copy partial result to registers
         {
            InterpHelperFunctions::ReadSliceFromShared< dof_shape, quad_shape >( thread, sy, y, std::tie( slice_index... ) );
         }
      });
   });

   // results of the shared contraction were written to register_buffer2, which
   // is now the input to the register contractions, so we swap them
   Swap(register_buffer1, register_buffer2);

// contraction on register dimensions
   using reg_input_shape = subsequence_t< dof_shape, RegisterDimensions >;
   using reg_output_shape = subsequence_t< quad_shape, RegisterDimensions >;

   // TODO Do a recursive call to allow using RecursiveArray instead of FixedFIFOView
   ConstexprLoop< RegisterBlockDim >( [&] ( auto _c )
   {
      constexpr Integer c = _c; // NVCC fails to correctly capture arguments to three or more nested lambda expression, so this line is necessary
      constexpr Integer ActiveDim = seq_get_v<c, RegisterDimensions>;

      using Op1D = std::tuple_element_t< ActiveDim, ProductOperator >;
      auto& B = std::get< ActiveDim >( element_quad_data );

      using contracted_dims = std::make_index_sequence< c >;
      using input_shape = replace_subsequence_t< reg_input_shape, reg_output_shape, contracted_dims >;
      
      using output_contracted_dims = std::make_index_sequence< c+1 >;
      using output_shape = replace_subsequence_t< reg_input_shape, reg_output_shape, output_contracted_dims >;

      auto rx = MakeFixedFIFOView( register_buffer1, input_shape{} );
      auto ry = MakeFixedFIFOView( register_buffer2, output_shape{} );

      ThreadLoop< tshape >( thread, [&] ( auto ... t )
      {
         // technically only some threads should perform the contractions (hence
         // the ThreadLoop) but there are no side effects if all the threads
         // perform the contractions since they are entirely in registers.
         // Should we get rid of the ThreadLoop?
         constexpr bool gradient = ActiveDim == DiffDim;
         InterpHelperFunctions::InterpContractionRegisters< gradient, c >( rx, ry, B, output_shape{} );
      });

      Swap( register_buffer1, register_buffer2 );
   });

   thread.SharedAllocator.reset();

   // return MakeFixedFIFOView( register_buffer1, reg_output_shape{} );
   // TODO1 copy result in serial recursive array? reg_output_shape + Loop?
   // TODO2 copy result in threaded recursive array? DofShape + Thread Loop?
   auto tmp_out = MakeFixedFIFOView( register_buffer1, reg_output_shape{} );
   auto result = MakeStaticFIFOView< Real >( reg_output_shape{} );
   UnitLoop< reg_output_shape >( [&]( auto ... indices )
   {
      result( indices... ) = tmp_out( indices... );
   });
   return result;
}

} // namespace gendil
