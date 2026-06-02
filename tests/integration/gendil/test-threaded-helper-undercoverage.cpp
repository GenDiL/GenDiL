// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <array>
#include <cmath>
#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-threaded-helper-undercoverage skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{
constexpr Real tolerance = 1.0e-10;

enum SegmentId : Integer
{
   thread_loop_segment = 0,
   quadrature_loop_segment,
   read_dofs_segment,
   write_dofs_segment,
   interpolate_values_segment,
   interpolate_gradient_segment,
   values_and_gradients_values_segment,
   values_and_gradients_gradient_segment,
   apply_test_functions_segment,
   apply_gradient_test_functions_segment,
   apply_values_and_gradient_test_functions_segment,
   num_segments
};

struct SegmentInfo
{
   const char * name;
   GlobalIndex offset;
   GlobalIndex entries_per_item;
   Integer components;
};

struct SegmentOffsets
{
   GlobalIndex offsets[ num_segments + 1 ];

   GENDIL_HOST_DEVICE
   GlobalIndex operator[]( const SegmentId segment ) const
   {
      return offsets[ static_cast< Integer >( segment ) ];
   }
};

template < typename Sequence >
struct SequenceProduct;

template < typename T, T... Values >
struct SequenceProduct< std::integer_sequence< T, Values... > >
{
   static constexpr GlobalIndex value =
      ( GlobalIndex{ 1 } * ... * static_cast< GlobalIndex >( Values ) );
};

template < typename Sequence >
inline constexpr GlobalIndex sequence_product_v =
   SequenceProduct< Sequence >::value;

template < typename Sequence >
struct SequenceFirst;

template < typename T, T First, T... Rest >
struct SequenceFirst< std::integer_sequence< T, First, Rest... > >
{
   static constexpr GlobalIndex value = static_cast< GlobalIndex >( First );
};

template < typename Sequence >
inline constexpr GlobalIndex sequence_first_v = SequenceFirst< Sequence >::value;

template < typename T, T... Sizes, typename... Indices >
GENDIL_HOST_DEVICE
GlobalIndex FlatIndex(
   std::integer_sequence< T, Sizes... >,
   Indices... indices )
{
   static_assert(
      sizeof...( Sizes ) == sizeof...( Indices ),
      "Wrong number of indices for FlatIndex." );

   const GlobalIndex idx[] = { static_cast< GlobalIndex >( indices )... };
   const GlobalIndex sizes[] = { static_cast< GlobalIndex >( Sizes )... };

   GlobalIndex linear = 0;
   for ( Integer i = 0; i < static_cast< Integer >( sizeof...( Sizes ) ); ++i )
   {
      linear = linear * sizes[ i ] + idx[ i ];
   }
   return linear;
}

template < typename T, typename IndexType, IndexType... Dims >
GENDIL_HOST_DEVICE
auto MakeElementTensorView(
   const HostDevicePointer< T > & data,
   std::integer_sequence< IndexType, Dims... >,
   const GlobalIndex num_items )
{
   return MakeFIFOView(
      data,
      static_cast< GlobalIndex >( Dims )...,
      num_items );
}

template < typename T, typename IndexType, IndexType... Dims >
GENDIL_HOST_DEVICE
auto MakeElementTensorView(
   T * data,
   std::integer_sequence< IndexType, Dims... >,
   const GlobalIndex num_items )
{
   return MakeFIFOView(
      data,
      static_cast< GlobalIndex >( Dims )...,
      num_items );
}

GENDIL_HOST_DEVICE
Real InputDofValue(
   const GlobalIndex item,
   const GlobalIndex dof_index )
{
   return 0.125 +
      0.37 * static_cast< Real >( item + 1 ) +
      0.071 * static_cast< Real >( dof_index + 1 ) +
      0.003 * static_cast< Real >( ( item + 1 ) * ( dof_index + 2 ) );
}

GENDIL_HOST_DEVICE
Real LocalDofValue(
   const GlobalIndex item,
   const GlobalIndex dof_index )
{
   return -0.4 +
      0.19 * static_cast< Real >( item + 1 ) +
      0.113 * static_cast< Real >( dof_index + 3 );
}

GENDIL_HOST_DEVICE
Real QuadValue(
   const GlobalIndex item,
   const GlobalIndex quad_index )
{
   return 0.25 +
      0.23 * static_cast< Real >( item + 1 ) -
      0.047 * static_cast< Real >( quad_index + 1 ) +
      0.002 * static_cast< Real >( ( item + 3 ) * ( quad_index + 1 ) );
}

GENDIL_HOST_DEVICE
Real GradientValue(
   const GlobalIndex item,
   const GlobalIndex quad_index,
   const Integer component )
{
   return -0.5 +
      0.17 * static_cast< Real >( item + 2 ) +
      0.031 * static_cast< Real >( quad_index + 1 ) +
      0.29 * static_cast< Real >( component + 1 );
}

void PrintAuditNote()
{
   std::cout
      << "Under-threaded helper coverage audit:\n"
      << "  ThreadLoop and QuadraturePointLoop: expected to support "
      << "strided coverage when num_threads_1D < loop extent.\n"
      << "  ReadDofs/WriteDofs: suspected first-dimension ordering "
      << "assumptions; diagnostics compare owned L2 cell paths only.\n"
      << "  InterpolateValues/InterpolateGradient/ApplyTestFunctions/"
      << "ApplyGradientTestFunctions: shared-memory reset and staging "
      << "sensitivity are intentionally exercised by running helpers in "
      << "sequence.\n"
      << "  InterpolateValuesAndGradients: diagnostic stores the paired "
      << "value and gradient stages explicitly, without changing production "
      << "wrapper behavior.\n"
      << "  AggregateDimensions: split into a separate diagnostic because it "
      << "has a distinct shared-accumulator initialization issue.\n";
}

Integer GetDeviceSharedMemoryPerBlockBytes()
{
#if defined( GENDIL_USE_CUDA )
   int device = 0;
   cudaDeviceProp properties;
   GENDIL_GPU_CHECK( cudaGetDevice( &device ) );
   GENDIL_GPU_CHECK( cudaGetDeviceProperties( &properties, device ) );
   return static_cast< Integer >( properties.sharedMemPerBlock );
#elif defined( GENDIL_USE_HIP )
   int device = 0;
   hipDeviceProp_t properties;
   GENDIL_GPU_CHECK( hipGetDevice( &device ) );
   GENDIL_GPU_CHECK( hipGetDeviceProperties( &properties, device ) );
   return static_cast< Integer >( properties.sharedMemPerBlock );
#else
   return 0;
#endif
}

template <
   typename Config,
   typename IntegrationRule >
bool SharedMemoryConfigurationFits( const char * label )
{
   using AggregateRule =
      decltype( GetSubIntegrationRule< 0, 1 >( IntegrationRule{} ) );
   constexpr size_t required_shared_mem =
      required_shared_memory_v< Config, IntegrationRule > +
      required_shared_memory_v< Config, AggregateRule >;
   constexpr size_t block_reals =
      KernelContext< Config, required_shared_mem >::shared_memory_block_size;
   constexpr size_t requested_bytes = block_reals * sizeof( Real );
   const Integer max_bytes = GetDeviceSharedMemoryPerBlockBytes();

   if ( requested_bytes > static_cast< size_t >( max_bytes ) )
   {
      std::cout << "Skipping " << label
                << ": requested " << requested_bytes
                << " bytes of shared memory per block, device limit is "
                << max_bytes << ".\n";
      return false;
   }
   return true;
}

template < typename KernelContext, typename Shape >
GENDIL_HOST_DEVICE
void CountThreadLoopVisits(
   KernelContext & kernel_conf,
   Real * output,
   const GlobalIndex base )
{
   ThreadLoop< Shape >(
      kernel_conf,
      [&] ( auto... indices )
      {
         const GlobalIndex index = FlatIndex( Shape{}, indices... );
         gendil::AtomicAdd( output[ base + index ], Real{ 1.0 } );
      } );
}

template < typename Rule, typename KernelContext >
GENDIL_HOST_DEVICE
void CountQuadratureLoopVisits(
   KernelContext & kernel_conf,
   Real * output,
   const GlobalIndex base )
{
   using QuadShape = typename Rule::points::num_points_tensor;
   QuadraturePointLoop< Rule >(
      kernel_conf,
      [&] ( const auto & quad_index )
      {
         GlobalIndex index = 0;
         ConstexprLoop< QuadShape::size() >(
            [&] ( auto i )
            {
               constexpr Integer I = decltype( i )::value;
               if constexpr ( I == 0 )
               {
                  index = quad_index[ I ];
               }
               else
               {
                  constexpr GlobalIndex dim =
                     static_cast< GlobalIndex >( seq_get_v< I, QuadShape > );
                  index = index * dim + quad_index[ I ];
               }
            } );
         gendil::AtomicAdd( output[ base + index ], Real{ 1.0 } );
      } );
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor >
GENDIL_HOST_DEVICE
void StoreDofSlice(
   KernelContext & kernel_conf,
   const GlobalIndex item,
   const LocalTensor & local,
   Real * output,
   const GlobalIndex base )
{
   using DofShape = orders_to_num_dofs<
      typename FiniteElementSpace::finite_element_type::
         shape_functions::orders >;
   using ThreadShape = subsequence_t<
      DofShape,
      typename KernelContext::template threaded_dimensions<
         DofShape::size() > >;
   using RegisterShape = subsequence_t<
      DofShape,
      typename KernelContext::template register_dimensions<
         DofShape::size() > >;
   constexpr GlobalIndex dof_size = sequence_product_v< DofShape >;

   ThreadLoop< ThreadShape >(
      kernel_conf,
      [&] ( auto... t )
      {
         UnitLoop< RegisterShape >(
            [&] ( auto... k )
            {
               const GlobalIndex local_index =
                  FlatIndex( DofShape{}, t..., k... );
               output[ base + item * dof_size + local_index ] = local( k... );
            } );
      } );
}

template < typename Rule, typename KernelContext, typename LocalTensor >
GENDIL_HOST_DEVICE
void StoreQuadSlice(
   KernelContext & kernel_conf,
   const GlobalIndex item,
   const LocalTensor & local,
   Real * output,
   const GlobalIndex base )
{
   using QuadShape = typename Rule::points::num_points_tensor;
   using ThreadShape = subsequence_t<
      QuadShape,
      typename KernelContext::template threaded_dimensions<
         QuadShape::size() > >;
   using RegisterShape = subsequence_t<
      QuadShape,
      typename KernelContext::template register_dimensions<
         QuadShape::size() > >;
   constexpr GlobalIndex quad_size = sequence_product_v< QuadShape >;

   ThreadLoop< ThreadShape >(
      kernel_conf,
      [&] ( auto... t )
      {
         UnitLoop< RegisterShape >(
            [&] ( auto... k )
            {
               const GlobalIndex local_index =
                  FlatIndex( QuadShape{}, t..., k... );
               output[ base + item * quad_size + local_index ] = local( k... );
            } );
      } );
}

template < typename Rule, typename KernelContext, typename LocalTensor >
GENDIL_HOST_DEVICE
void StoreGradientSlice(
   KernelContext & kernel_conf,
   const GlobalIndex item,
   const LocalTensor & local,
   Real * output,
   const GlobalIndex base )
{
   using QuadShape = typename Rule::points::num_points_tensor;
   using ThreadShape = subsequence_t<
      QuadShape,
      typename KernelContext::template threaded_dimensions<
         QuadShape::size() > >;
   using RegisterShape = subsequence_t<
      QuadShape,
      typename KernelContext::template register_dimensions<
         QuadShape::size() > >;
   constexpr Integer dim = QuadShape::size();
   constexpr GlobalIndex quad_size = sequence_product_v< QuadShape >;

   ThreadLoop< ThreadShape >(
      kernel_conf,
      [&] ( auto... t )
      {
         UnitLoop< RegisterShape >(
            [&] ( auto... k )
            {
               const GlobalIndex quad_index =
                  FlatIndex( QuadShape{}, t..., k... );
               ConstexprLoop< dim >(
                  [&] ( auto component )
                  {
                     constexpr Integer c = decltype( component )::value;
                     output[
                        base +
                        item * quad_size * dim +
                        quad_index * dim +
                        c ] = local( k..., c );
                  } );
            } );
      } );
}

template < typename Rule, typename KernelContext, typename LocalTensor >
GENDIL_HOST_DEVICE
void FillQuadValues(
   KernelContext & kernel_conf,
   const GlobalIndex item,
   LocalTensor & local )
{
   using QuadShape = typename Rule::points::num_points_tensor;
   using ThreadShape = subsequence_t<
      QuadShape,
      typename KernelContext::template threaded_dimensions<
         QuadShape::size() > >;
   using RegisterShape = subsequence_t<
      QuadShape,
      typename KernelContext::template register_dimensions<
         QuadShape::size() > >;

   ThreadLoop< ThreadShape >(
      kernel_conf,
      [&] ( auto... t )
      {
         UnitLoop< RegisterShape >(
            [&] ( auto... k )
            {
               const GlobalIndex quad_index =
                  FlatIndex( QuadShape{}, t..., k... );
               local( k... ) = QuadValue( item, quad_index );
            } );
      } );
}

template < typename Rule, typename KernelContext, typename LocalTensor >
GENDIL_HOST_DEVICE
void FillGradientValues(
   KernelContext & kernel_conf,
   const GlobalIndex item,
   LocalTensor & local )
{
   using QuadShape = typename Rule::points::num_points_tensor;
   using ThreadShape = subsequence_t<
      QuadShape,
      typename KernelContext::template threaded_dimensions<
         QuadShape::size() > >;
   using RegisterShape = subsequence_t<
      QuadShape,
      typename KernelContext::template register_dimensions<
         QuadShape::size() > >;
   constexpr Integer dim = QuadShape::size();

   ThreadLoop< ThreadShape >(
      kernel_conf,
      [&] ( auto... t )
      {
         UnitLoop< RegisterShape >(
            [&] ( auto... k )
            {
               const GlobalIndex quad_index =
                  FlatIndex( QuadShape{}, t..., k... );
               ConstexprLoop< dim >(
                  [&] ( auto component )
                  {
                     constexpr Integer c = decltype( component )::value;
                     local( k..., c ) =
                        GradientValue( item, quad_index, c );
                  } );
            } );
      } );
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor >
GENDIL_HOST_DEVICE
void FillDofValues(
   KernelContext & kernel_conf,
   const GlobalIndex item,
   LocalTensor & local )
{
   using DofShape = orders_to_num_dofs<
      typename FiniteElementSpace::finite_element_type::
         shape_functions::orders >;
   using ThreadShape = subsequence_t<
      DofShape,
      typename KernelContext::template threaded_dimensions<
         DofShape::size() > >;
   using RegisterShape = subsequence_t<
      DofShape,
      typename KernelContext::template register_dimensions<
         DofShape::size() > >;

   ThreadLoop< ThreadShape >(
      kernel_conf,
      [&] ( auto... t )
      {
         UnitLoop< RegisterShape >(
            [&] ( auto... k )
            {
               const GlobalIndex dof_index =
                  FlatIndex( DofShape{}, t..., k... );
               local( k... ) = LocalDofValue( item, dof_index );
            } );
      } );
}

template <
   typename Config,
   typename FiniteElementSpace,
   typename IntegrationRule >
void RunHelperCoverageKernel(
   const FiniteElementSpace & fe_space,
   const IntegrationRule &,
   const GlobalIndex num_items,
   const SegmentOffsets segment_offsets,
   const HostDevicePointer< Real > & input_data,
   HostDevicePointer< Real > output_data )
{
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;
   using DofShape = orders_to_num_dofs<
      typename ShapeFunctions::orders >;
   using QuadShape = typename IntegrationRule::points::num_points_tensor;
   using HelperShape = max_sequence_t< DofShape, QuadShape >;

   constexpr size_t required_shared_mem =
      required_shared_memory_v< Config, IntegrationRule >;
   using Context = KernelContext< Config, required_shared_mem >;
   static constexpr bool dof_supported =
      threaded_shape_covered_v< Config, DofShape >;
   static constexpr bool helper_supported =
      threaded_shape_covered_v< Config, HelperShape >;

   const auto element_quad_data =
      MakeDofToQuad< ShapeFunctions, IntegrationRule >();

   Config::CandidateBlockLoop(
      num_items,
      [=] GENDIL_HOST_DEVICE () mutable
      {
         GENDIL_SHARED Real _shared_mem[
            Context::shared_memory_block_size ];
         Context kernel_conf( _shared_mem );

         const GlobalIndex item = Config::WorkItemIndex();
         Real * output = output_data;
         auto input_view =
            MakeElementTensorView( input_data, DofShape{}, num_items );
         auto write_view =
            MakeElementTensorView(
               output + segment_offsets[ write_dofs_segment ],
               DofShape{},
               num_items );

         CountThreadLoopVisits< Context, QuadShape >(
            kernel_conf,
            output,
            segment_offsets[ thread_loop_segment ] +
               item * sequence_product_v< QuadShape > );
         CountQuadratureLoopVisits< IntegrationRule >(
            kernel_conf,
            output,
            segment_offsets[ quadrature_loop_segment ] +
               item * sequence_product_v< QuadShape > );

         using RegisterDofShape = subsequence_t<
            DofShape,
            typename Config::template register_dimensions<
               DofShape::size() > >;
         using RegisterQuadShape = subsequence_t<
            QuadShape,
            typename Config::template register_dimensions<
               QuadShape::size() > >;
         using RegisterGradientShape =
            cat_t< RegisterQuadShape, std::index_sequence< QuadShape::size() > >;

         // CUDA fix
         (void)fe_space;
         (void)element_quad_data;

         if constexpr ( dof_supported )
         {
            auto local_read =
               ReadDofs( kernel_conf, fe_space, item, input_view );
            StoreDofSlice< Context, FiniteElementSpace >(
                  kernel_conf,
                  item,
                  local_read,
                  output,
                  segment_offsets[ read_dofs_segment ] );

            auto local_write =
               MakeStaticFIFOView< Real >( RegisterDofShape{} );
            FillDofValues< Context, FiniteElementSpace >(
                  kernel_conf,
                  item,
                  local_write );
            WriteDofs( kernel_conf, fe_space, item, local_write, write_view );

            if constexpr ( helper_supported )
            {
               auto interpolated_values =
                  InterpolateValues(
                     kernel_conf,
                     element_quad_data,
                     local_read );
               StoreQuadSlice< IntegrationRule >(
                  kernel_conf,
                  item,
                  interpolated_values,
                  output,
                  segment_offsets[ interpolate_values_segment ] );

               auto interpolated_gradient =
                  InterpolateGradient(
                     kernel_conf,
                     element_quad_data,
                     local_read );
               StoreGradientSlice< IntegrationRule >(
                  kernel_conf,
                  item,
                  interpolated_gradient,
                  output,
                  segment_offsets[ interpolate_gradient_segment ] );

               StoreQuadSlice< IntegrationRule >(
                  kernel_conf,
                  item,
                  interpolated_values,
                  output,
                  segment_offsets[ values_and_gradients_values_segment ] );
               StoreGradientSlice< IntegrationRule >(
                  kernel_conf,
                  item,
                  interpolated_gradient,
                  output,
                  segment_offsets[ values_and_gradients_gradient_segment ] );
            }
         }

         if constexpr ( helper_supported )
         {
            auto quad_input = MakeStaticFIFOView< Real >( RegisterQuadShape{} );
            FillQuadValues< IntegrationRule >( kernel_conf, item, quad_input );

            auto applied_test =
               ApplyTestFunctions(
                  kernel_conf,
                  element_quad_data,
                  quad_input );
            StoreDofSlice< Context, FiniteElementSpace >(
                  kernel_conf,
                  item,
                  applied_test,
                  output,
                  segment_offsets[ apply_test_functions_segment ] );

            auto gradient_input =
               MakeStaticFIFOView< Real >( RegisterGradientShape{} );
            FillGradientValues< IntegrationRule >(
               kernel_conf,
               item,
               gradient_input );

            auto gradient_test_q =
               ApplyGradientTestFunctionsAtQPoints(
                  kernel_conf,
                  element_quad_data,
                  gradient_input );
            auto applied_gradient =
               ApplyTestFunctions(
                  kernel_conf,
                  element_quad_data,
                  gradient_test_q );
            StoreDofSlice< Context, FiniteElementSpace >(
                  kernel_conf,
                  item,
                  applied_gradient,
                  output,
                  segment_offsets[ apply_gradient_test_functions_segment ] );

            auto values_and_gradient_test_q =
               ApplyGradientTestFunctionsAtQPoints(
                  kernel_conf,
                  element_quad_data,
                  gradient_input );
            values_and_gradient_test_q += quad_input;
            auto applied_values_and_gradient =
               ApplyTestFunctions(
                  kernel_conf,
                  element_quad_data,
                  values_and_gradient_test_q );
            StoreDofSlice< Context, FiniteElementSpace >(
                  kernel_conf,
                  item,
                  applied_values_and_gradient,
                  output,
                  segment_offsets[
                     apply_values_and_gradient_test_functions_segment ] );
         }
      } );
   GENDIL_DEVICE_SYNC;
}

template < typename FiniteElementSpace, typename IntegrationRule >
std::array< SegmentInfo, num_segments > MakeSegmentInfo(
   const GlobalIndex num_items )
{
   using DofShape = orders_to_num_dofs<
      typename FiniteElementSpace::finite_element_type::
         shape_functions::orders >;
   using QuadShape = typename IntegrationRule::points::num_points_tensor;

   constexpr GlobalIndex dof_size = sequence_product_v< DofShape >;
   constexpr GlobalIndex quad_size = sequence_product_v< QuadShape >;
   constexpr Integer dim = QuadShape::size();

   std::array< SegmentInfo, num_segments > segments{ {
      { "ThreadLoop", 0, quad_size, 1 },
      { "QuadraturePointLoop", 0, quad_size, 1 },
      { "ReadDofs", 0, dof_size, 1 },
      { "WriteDofs", 0, dof_size, 1 },
      { "InterpolateValues", 0, quad_size, 1 },
      { "InterpolateGradient", 0, quad_size * dim, dim },
      { "InterpolateValuesAndGradients values", 0, quad_size, 1 },
      { "InterpolateValuesAndGradients gradients", 0, quad_size * dim, dim },
      { "ApplyTestFunctions", 0, dof_size, 1 },
      { "ApplyGradientTestFunctions", 0, dof_size, 1 },
      { "ApplyValuesAndGradientTestFunctions", 0, dof_size, 1 }
   } };

   GlobalIndex offset = 0;
   for ( auto & segment : segments )
   {
      segment.offset = offset;
      offset += segment.entries_per_item * num_items;
   }
   return segments;
}

SegmentOffsets MakeSegmentOffsets(
   const std::array< SegmentInfo, num_segments > & segments )
{
   SegmentOffsets offsets{};
   for ( Integer i = 0; i < num_segments; ++i )
   {
      offsets.offsets[ i ] = segments[ i ].offset;
   }
   const auto & last = segments[ num_segments - 1 ];
   offsets.offsets[ num_segments ] =
      last.offset + last.entries_per_item;
   return offsets;
}

GlobalIndex TotalOutputSize(
   const std::array< SegmentInfo, num_segments > & segments,
   const GlobalIndex num_items )
{
   const auto & last = segments[ num_segments - 1 ];
   return last.offset + last.entries_per_item * num_items;
}

void InitializeInput(
   DeviceBuffer< Real > & input,
   const GlobalIndex dof_size,
   const GlobalIndex num_items )
{
   for ( GlobalIndex item = 0; item < num_items; ++item )
   {
      for ( GlobalIndex dof = 0; dof < dof_size; ++dof )
      {
         input.data.host_pointer[ item * dof_size + dof ] =
            InputDofValue( item, dof );
      }
   }
   ToDevice( input.size, input.data );
}

template <
   typename Config,
   typename FiniteElementSpace,
   typename IntegrationRule >
bool IsSegmentSupported( const SegmentId segment )
{
   using DofShape = orders_to_num_dofs<
      typename FiniteElementSpace::finite_element_type::
         shape_functions::orders >;
   using QuadShape = typename IntegrationRule::points::num_points_tensor;
   using HelperShape = max_sequence_t< DofShape, QuadShape >;

   static constexpr bool dof_supported =
      threaded_shape_covered_v< Config, DofShape >;
   static constexpr bool helper_supported =
      threaded_shape_covered_v< Config, HelperShape >;

   switch ( segment )
   {
      case thread_loop_segment:
      case quadrature_loop_segment:
         return true;
      case read_dofs_segment:
      case write_dofs_segment:
         return dof_supported;
      default:
         return helper_supported;
   }
}

bool CheckSegment(
   const char * label,
   const SegmentInfo & segment,
   const GlobalIndex num_items,
   DeviceBuffer< Real > & observed,
   DeviceBuffer< Real > & expected )
{
   bool success = true;
   const GlobalIndex total_entries =
      segment.entries_per_item * num_items;
   const Integer components = std::max( Integer{ 1 }, segment.components );

   for ( Integer component = 0; component < components; ++component )
   {
      Real err_sq = 0.0;
      Real norm_sq = 0.0;
      Real observed_sum = 0.0;
      Real expected_sum = 0.0;
      Real observed_sumsq = 0.0;
      Real expected_sumsq = 0.0;
      Real max_abs = 0.0;
      GlobalIndex mismatches = 0;
      GlobalIndex first_mismatch = 0;
      bool has_first_mismatch = false;

      for ( GlobalIndex i = 0; i < total_entries; ++i )
      {
         if ( segment.components > 1 &&
              i % segment.components != component )
         {
            continue;
         }

         const GlobalIndex index = segment.offset + i;
         const Real obs = observed.data.host_pointer[ index ];
         const Real exp = expected.data.host_pointer[ index ];
         const Real diff = obs - exp;

         err_sq += diff * diff;
         norm_sq += exp * exp;
         observed_sum += obs;
         expected_sum += exp;
         observed_sumsq += obs * obs;
         expected_sumsq += exp * exp;
         max_abs = std::max( max_abs, std::abs( obs ) );

         const Real scale = std::max( Real{ 1.0 }, std::abs( exp ) );
         if ( !std::isfinite( obs ) ||
              !std::isfinite( exp ) ||
              std::abs( diff ) / scale > tolerance )
         {
            ++mismatches;
            if ( !has_first_mismatch )
            {
               first_mismatch = i;
               has_first_mismatch = true;
            }
         }
      }

      const Real scaled_error =
         std::sqrt( err_sq ) / std::max( Real{ 1.0 }, std::sqrt( norm_sq ) );
      std::cout
         << label << " " << segment.name;
      if ( segment.components > 1 )
      {
         std::cout << ", component " << component;
      }
      std::cout
         << ": scaled L2 error = " << scaled_error
         << ", observed(sum=" << observed_sum
         << ", sumsq=" << observed_sumsq
         << ", maxabs=" << max_abs
         << "), expected(sum=" << expected_sum
         << ", sumsq=" << expected_sumsq
         << "), mismatches=" << mismatches;
      if ( has_first_mismatch )
      {
         const GlobalIndex index = segment.offset + first_mismatch;
         std::cout
            << ", first mismatch local-entry=" << first_mismatch
            << ", observed=" << observed.data.host_pointer[ index ]
            << ", expected=" << expected.data.host_pointer[ index ];
      }
      std::cout << ".\n";

      if ( !std::isfinite( scaled_error ) ||
           scaled_error > tolerance ||
           mismatches > 0 )
      {
         success = false;
      }
   }

   return success;
}

template <
   typename Config,
   typename FiniteElementSpace,
   typename IntegrationRule >
void RunAllHelpers(
   const FiniteElementSpace & fe_space,
   const IntegrationRule & integration_rule,
   const GlobalIndex num_items,
   const SegmentOffsets segment_offsets,
   DeviceBuffer< Real > & output )
{
   using DofShape = orders_to_num_dofs<
      typename FiniteElementSpace::finite_element_type::
         shape_functions::orders >;
   constexpr GlobalIndex dof_size = sequence_product_v< DofShape >;

   DeviceBuffer< Real > input( dof_size * num_items, 0.0 );
   InitializeInput( input, dof_size, num_items );

   RunHelperCoverageKernel< Config >(
      fe_space,
      integration_rule,
      num_items,
      segment_offsets,
      input.data,
      output.data );
   output.CopyToHost();
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FiniteElementSpace,
   typename IntegrationRule >
bool RunConfigurationCase(
   const char * label,
   const FiniteElementSpace & fe_space,
   const IntegrationRule & integration_rule )
{
   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return true;
   }

   using Config =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;
   using OracleConfig =
      DeviceKernelConfiguration< ThreadBlockLayout<>, 0, 1 >;

   if ( !SharedMemoryConfigurationFits< Config, IntegrationRule >( label ) )
   {
      return true;
   }

   const GlobalIndex num_items = BatchSize == 1 ? 3 : BatchSize;
   auto segments =
      MakeSegmentInfo< FiniteElementSpace, IntegrationRule >( num_items );
   const SegmentOffsets segment_offsets = MakeSegmentOffsets( segments );
   const GlobalIndex total_output_size =
      TotalOutputSize( segments, num_items );

   std::cout
      << "Under-threaded helper case: " << label
      << ", num_items = " << num_items
      << ", thread_block_dim = " << Layout::thread_block_dim
      << ", logical_threads_per_work_item = "
      << Layout::GetNumberOfThreads()
      << ", BatchSize = " << BatchSize << ".\n";

   DeviceBuffer< Real > expected( total_output_size, 0.0 );
   DeviceBuffer< Real > observed( total_output_size, 0.0 );

   RunAllHelpers< OracleConfig >(
      fe_space,
      integration_rule,
      num_items,
      segment_offsets,
      expected );
   RunAllHelpers< Config >(
      fe_space,
      integration_rule,
      num_items,
      segment_offsets,
      observed );

   bool success = true;
   for ( Integer i = 0; i < num_segments; ++i )
   {
      const SegmentId segment_id = static_cast< SegmentId >( i );
      const auto & segment = segments[ i ];
      if ( !IsSegmentSupported< Config, FiniteElementSpace, IntegrationRule >(
              segment_id ) )
      {
         std::cout
            << label << " " << segment.name
            << ": EXPECTED_UNSUPPORTED - Under-threaded strided coverage "
            << "is not supported by this threaded helper yet.\n";
         continue;
      }

      success =
         CheckSegment(
            label,
            segment,
            num_items,
            observed,
            expected ) && success;
   }
   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   Integer... Orders,
   Integer... NumQuads >
bool RunUndercoverageCase(
   const char * label,
   std::integer_sequence< Integer, Orders... >,
   std::integer_sequence< Integer, NumQuads... > )
{
   static_assert( sizeof...( Orders ) == sizeof...( NumQuads ) );
   constexpr Integer dim = sizeof...( Orders );
   constexpr GlobalIndex num_items = BatchSize == 1 ? 3 : BatchSize;

   FiniteElementOrders< Orders... > orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   IntegrationRuleNumPoints< NumQuads... > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   if constexpr ( dim == 1 )
   {
      Cartesian1DMesh mesh( 1.0 / static_cast< Real >( num_items ), num_items );
      auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
      return RunConfigurationCase< Layout, MaxSharedDimensions, BatchSize >(
         label,
         fe_space,
         integration_rule );
   }
   else if constexpr ( dim == 2 )
   {
      Cartesian1DMesh mesh_x(
         1.0 / static_cast< Real >( num_items ),
         num_items );
      Cartesian1DMesh mesh_y( 1.0, 1 );
      auto mesh = MakeCartesianProductMesh( mesh_x, mesh_y );
      auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
      return RunConfigurationCase< Layout, MaxSharedDimensions, BatchSize >(
         label,
         fe_space,
         integration_rule );
   }
   else
   {
      static_assert( dim == 3 );
      Cartesian1DMesh mesh_x(
         1.0 / static_cast< Real >( num_items ),
         num_items );
      Cartesian1DMesh mesh_y( 1.0, 1 );
      Cartesian1DMesh mesh_z( 1.0, 1 );
      auto mesh = MakeCartesianProductMesh( mesh_x, mesh_y, mesh_z );
      auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
      return RunConfigurationCase< Layout, MaxSharedDimensions, BatchSize >(
         label,
         fe_space,
         integration_rule );
   }
}

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool Run1DCase( const char * label )
{
   return RunUndercoverageCase< Layout, MaxSharedDimensions, BatchSize >(
      label,
      std::integer_sequence< Integer, 3 >{},
      std::integer_sequence< Integer, 5 >{} );
}

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool Run2DCase( const char * label )
{
   return RunUndercoverageCase< Layout, MaxSharedDimensions, BatchSize >(
      label,
      std::integer_sequence< Integer, 3, 2 >{},
      std::integer_sequence< Integer, 5, 7 >{} );
}

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool Run3DCase( const char * label )
{
   return RunUndercoverageCase< Layout, MaxSharedDimensions, BatchSize >(
      label,
      std::integer_sequence< Integer, 2, 3, 1 >{},
      std::integer_sequence< Integer, 4, 5, 3 >{} );
}

template < Integer BatchSize >
bool Run1DMatrix()
{
   bool success = true;
   success = Run1DCase< ThreadBlockLayout< 1 >, 1, BatchSize >(
      "1D ThreadBlockLayout<1>" ) && success;
   success = Run1DCase< ThreadBlockLayout< 2 >, 1, BatchSize >(
      "1D ThreadBlockLayout<2>" ) && success;
   success = Run1DCase< ThreadBlockLayout< 3 >, 1, BatchSize >(
      "1D ThreadBlockLayout<3>" ) && success;
   success = Run1DCase< ThreadBlockLayout< 4 >, 1, BatchSize >(
      "1D ThreadBlockLayout<4>" ) && success;
   success = Run1DCase< ThreadBlockLayout< 5 >, 1, BatchSize >(
      "1D ThreadBlockLayout<5>" ) && success;
   return success;
}

template < Integer BatchSize >
bool Run2DMatrix()
{
   bool success = true;
   success = Run2DCase< ThreadBlockLayout< 1, 1 >, 2, BatchSize >(
      "2D ThreadBlockLayout<1,1>" ) && success;
   success = Run2DCase< ThreadBlockLayout< 2, 1 >, 2, BatchSize >(
      "2D ThreadBlockLayout<2,1>" ) && success;
   success = Run2DCase< ThreadBlockLayout< 1, 3 >, 2, BatchSize >(
      "2D ThreadBlockLayout<1,3>" ) && success;
   success = Run2DCase< ThreadBlockLayout< 2, 3 >, 2, BatchSize >(
      "2D ThreadBlockLayout<2,3>" ) && success;
   success = Run2DCase< ThreadBlockLayout< 4, 3 >, 2, BatchSize >(
      "2D ThreadBlockLayout<4,3>" ) && success;
   success = Run2DCase< ThreadBlockLayout< 5, 7 >, 2, BatchSize >(
      "2D ThreadBlockLayout<5,7>" ) && success;
   return success;
}

template < Integer BatchSize >
bool Run3DMatrix()
{
   bool success = true;
   success = Run3DCase< ThreadBlockLayout< 1, 1, 1 >, 3, BatchSize >(
      "3D ThreadBlockLayout<1,1,1>" ) && success;
   success = Run3DCase< ThreadBlockLayout< 2, 2, 1 >, 3, BatchSize >(
      "3D ThreadBlockLayout<2,2,1>" ) && success;
   success = Run3DCase< ThreadBlockLayout< 2, 3, 2 >, 3, BatchSize >(
      "3D ThreadBlockLayout<2,3,2>" ) && success;
   return success;
}

} // namespace

int main()
{
   PrintAuditNote();

   bool success = true;
   success = Run1DMatrix< 1 >() && success;
   success = Run1DMatrix< device_warp_size >() && success;
   success = Run2DMatrix< 1 >() && success;
   success = Run2DMatrix< device_warp_size >() && success;
   success = Run3DMatrix< 1 >() && success;
   success = Run3DMatrix< device_warp_size >() && success;

   if ( success )
   {
      std::cout
         << "SUMMARY: supported threaded helper diagnostics matched the "
         << "register-only oracle, and under-threaded helper cases were "
         << "classified as EXPECTED_UNSUPPORTED before invoking guarded "
         << "helper paths.\n";
   }
   else
   {
      std::cout
         << "SUMMARY: one or more under-threaded helper diagnostics "
         << "mismatched the register-only oracle. Use the segment/component "
         << "diagnostics above to choose the next helper-level fix.\n";
   }

   return success ? 0 : 1;
}

#endif
