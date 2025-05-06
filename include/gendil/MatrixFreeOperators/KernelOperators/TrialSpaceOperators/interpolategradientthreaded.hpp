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

namespace GradHelperFunctions
{
   // Contraction along shared dimension writing directly to register output
   template < Integer SharedDim, Integer ActiveDim, typename QuadShape, typename KernelContext, typename InputTensor, typename OutputTensor, typename Op1D, typename SliceIndexTuple >
   GENDIL_HOST_DEVICE
   void GradContractionShared( const KernelContext & thread, const InputTensor & sx, OutputTensor& Gu, const Op1D & B, SliceIndexTuple && slice_index )
   {
      constexpr Integer NQ = details::range_dim_v< Op1D >;

      using tshape = subsequence_t< QuadShape, typename KernelContext::template threaded_dimensions< QuadShape::size() > >;
      using rshape = subsequence_t< QuadShape, typename KernelContext::template shared_register_dimensions< QuadShape::size() > >;

      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape >( [&] ( auto... j )
         {
            auto sx_view = MakeDimensionSubView< SharedDim >( sx, t..., j... );
            
            const Integer p = get< SharedDim >( t..., j... );

            Real value = 0.0;

            for ( Integer q = 0; q < NQ; ++q )
            {
               const Real g = B.quad_gradients( p, q );
               value += g * sx_view( q );
            }

            Gu( j..., ActiveDim ) = value;
         });
      });
   }

   template < Integer RegisterDim, Integer ActiveDim, typename rshape, typename InputTensor, typename OutputTensor, typename Op1D >
   GENDIL_HOST_DEVICE
   void GradContractionRegisters( const InputTensor & x, OutputTensor & Gu, const Op1D & B )
   {
      constexpr Integer NQ = details::range_dim_v< Op1D >;

      Loop< rshape >( [&] ( auto... j )
      {
         auto x_view = MakeDimensionSubView< RegisterDim >( x, j... );
         const Integer p = get< RegisterDim >( j... );

         Real value = 0.0;

         for ( Integer q = 0; q < NQ; ++q )
         {
            const Real g = B.quad_gradients( p, q );
            value += g * x_view( q );
         }

         Gu( j..., ActiveDim ) = value;
      });
   }

   template < typename shape, typename KernelContext, typename RegisterTensor, typename SharedTensor, typename SliceIndexTuple >
   GENDIL_HOST_DEVICE
   void WriteToShared( const KernelContext & thread, const RegisterTensor & x, SharedTensor & sx, SliceIndexTuple&& slice_index )
   {
      using rshape_shared = subsequence_t< shape, typename KernelContext::template shared_register_dimensions< shape::size() > >;
      using tshape        = subsequence_t< shape, typename KernelContext::template threaded_dimensions< shape::size() > >;

      ThreadLoop< tshape >(thread, [&] ( auto... t )
      {
         UnitLoop< rshape_shared >( [&] ( auto... j )
         {
            auto idx = std::tuple_cat( std::tie( j... ), std::forward<SliceIndexTuple>( slice_index ) );
            sx( t..., j... ) = std::apply( x, idx );
         });
      });

      thread.Synchronize();
   }
} // namespace GradHelperFunctions

template < typename KernelContext, typename ProductOperator, typename InputTensor, typename OutputTensor >
GENDIL_HOST_DEVICE
void InterpolateGradientAtQPointsThreaded(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & uq,
   OutputTensor & Gu )
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
   constexpr size_t shared_buffer_size = shared_block_size_v< KernelContext, quad_shape >;
   Real * shared_data = thread.SharedAllocator.allocate( shared_buffer_size );
   auto sx = MakeFixedFIFOView( shared_data, shared_shape{} );

// contractions along threaded dimensions
   using rshape_non_shared = subsequence_t< quad_shape, NonSharedRegisterDimensions >;
   using rshape_shared = subsequence_t< quad_shape, SharedRegisterDimensions >;
   
   UnitLoop< rshape_non_shared >( [&] ( auto... slice_index )
   {
      // write to shared
      GradHelperFunctions::WriteToShared< quad_shape >( thread, uq, sx, std::tie( slice_index... ) );

      ConstexprLoop< ThreadBlockDim >( [&] ( auto _c )
      {
         constexpr Integer c = _c;
         constexpr Integer ActiveDim = seq_get_v< c, ThreadedDimensions >;

         using Op1D = std::tuple_element_t< ActiveDim, ProductOperator >;
         auto& B = std::get< ActiveDim >( element_quad_data );

         GradHelperFunctions::GradContractionShared< c, ActiveDim, quad_shape >( thread, sx, Gu, B, std::tie( slice_index... ) );

         if constexpr ( c+1 < ThreadBlockDim )
            thread.Synchronize();
      });
   });

// contraction along non-threaded dimensions
   using rshape = subsequence_t< quad_shape, RegisterDimensions >;
   using tshape = subsequence_t< quad_shape, ThreadedDimensions >;

   ConstexprLoop< RegisterBlockDim >( [&] ( auto _c )
   {
      constexpr Integer c = _c;
      constexpr Integer ActiveDim = seq_get_v< c, RegisterDimensions >;

      using Op1D = std::tuple_element_t< ActiveDim, ProductOperator >;
      auto& B = std::get< ActiveDim >( element_quad_data );

      ThreadLoop< tshape >( thread, [&] ( auto... )
      {
         GradHelperFunctions::GradContractionRegisters< c, ActiveDim, rshape >( uq, Gu, B );
      });
   });

   thread.SharedAllocator.reset();
}

template < typename KernelContext, typename ProductOperator, typename InputTensor, typename OutputTensor >
GENDIL_HOST_DEVICE
void InterpolateGradientThreaded(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & u,
   OutputTensor & Gu )
{
   constexpr bool face_interp = is_face_interpolation_v< ProductOperator >;

   if constexpr ( face_interp )
   {
      constexpr Integer Rank = std::tuple_size_v< ProductOperator >;
      ConstexprLoop< Rank >( [&] ( auto ActiveDim )
      {
         auto gu = InterpolateValuesThreaded< ActiveDim >( thread, element_quad_data, u );
         using quad_shape = make_contraction_output_shape< ProductOperator >;
         using RegisterDimensions = typename KernelContext::template register_dimensions< Rank >;
         using reg_shape = subsequence_t< quad_shape, RegisterDimensions >;
         UnitLoop< reg_shape >([&]( auto ... indices )
         {
            Gu( indices..., ActiveDim ) = gu( indices... );
         });
      } );
   }
   else
   {
      auto uq = InterpolateValuesThreaded( thread, element_quad_data, u );
      InterpolateGradientAtQPointsThreaded( thread, element_quad_data, uq, Gu );
   }
}

} // namespace gendil
