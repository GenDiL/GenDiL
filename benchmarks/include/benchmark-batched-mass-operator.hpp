// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "benchmark-common.hpp"

#include <iostream>
#include <type_traits>

namespace gendil::benchmarks
{

#if defined( GENDIL_USE_DEVICE )
constexpr GlobalIndex mass_target_dofs = 2'000'000;
constexpr GlobalIndex mass_min_cells = 4'096;
#else
constexpr GlobalIndex mass_target_dofs = 100'000;
constexpr GlobalIndex mass_min_cells = 512;
#endif

template < Integer Dim, Integer Order >
std::array< Integer, Dim > MassBenchmarkExtents()
{
   const GlobalIndex cells_from_dofs =
      mass_target_dofs / DofsPerElement< Dim, Order >();
   const GlobalIndex target_cells =
      cells_from_dofs > mass_min_cells ? cells_from_dofs : mass_min_cells;
   return BalancedExtents< Dim >( target_cells );
}

inline void PrintMassHeader()
{
   std::cout
      << "benchmark,dimension,order,num_quad_1d,num_cells,num_dofs,"
      << "num_qpts,kernel,layout,threaded_dimensions,"
      << "target_threads_per_block,threads_per_work_item,batch_size,"
      << "total_threads_per_block,block_x,block_y,block_z,"
      << "shared_memory_per_work_item,shared_memory_per_block,"
      << "time_per_apply,elements_per_s,dofs_per_s,qpts_per_s,status\n";
}

inline void PrintMassRow(
   const Integer dim,
   const Integer order,
   const Integer num_quad_1d,
   const GlobalIndex num_cells,
   const GlobalIndex num_dofs,
   const GlobalIndex num_qpts,
   const char * kernel,
   const char * layout,
   const size_t threaded_dimensions,
   const size_t target_threads_per_block,
   const size_t threads_per_work_item,
   const size_t batch_size,
   const std::array< GlobalIndex, 3 > & block_dims,
   const size_t shared_memory_per_work_item,
   const size_t shared_memory_per_block,
   const double time_per_apply,
   const char * status )
{
   const size_t total_threads_per_block =
      threads_per_work_item * batch_size;
   const double elements_per_s =
      time_per_apply > 0.0
         ? static_cast< double >( num_cells ) / time_per_apply
         : 0.0;
   const double dofs_per_s =
      time_per_apply > 0.0
         ? static_cast< double >( num_dofs ) / time_per_apply
         : 0.0;
   const double qpts_per_s =
      time_per_apply > 0.0
         ? static_cast< double >( num_qpts ) / time_per_apply
         : 0.0;

   std::cout
      << "batched-mass,"
      << dim << ','
      << order << ','
      << num_quad_1d << ','
      << num_cells << ','
      << num_dofs << ','
      << num_qpts << ','
      << kernel << ','
      << layout << ','
      << threaded_dimensions << ','
      << target_threads_per_block << ','
      << threads_per_work_item << ','
      << batch_size << ','
      << total_threads_per_block << ','
      << block_dims[ 0 ] << ','
      << block_dims[ 1 ] << ','
      << block_dims[ 2 ] << ','
      << shared_memory_per_work_item << ','
      << shared_memory_per_block << ','
      << time_per_apply << ','
      << elements_per_s << ','
      << dofs_per_s << ','
      << qpts_per_s << ','
      << status << '\n';
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
struct MassCaseSupported
{
   using dof_shape =
      orders_to_num_dofs<
         typename FiniteElementSpace::finite_element_type::
            shape_functions::orders >;
   using quad_shape =
      typename std::remove_cvref_t< IntegrationRule >::points::
         num_points_tensor;
   using helper_shape = max_sequence_t< dof_shape, quad_shape >;

   static constexpr bool value =
      threaded_shape_covered_v< KernelPolicy, dof_shape > &&
      threaded_shape_covered_v< KernelPolicy, helper_shape >;
};

template <
   Integer Dim,
   Integer Order,
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
void RunTimedMassRow(
   const FiniteElementSpace & fe_space,
   const IntegrationRule & integration_rule,
   const char * kernel_name,
   const char * layout_name,
   const size_t threaded_dimensions,
   const size_t target_threads_per_block,
   const size_t batch_size,
   const GlobalIndex num_cells,
   const GlobalIndex num_dofs,
   const GlobalIndex num_qpts,
   const std::array< GlobalIndex, 3 > & block_dims,
   const size_t required_shared_mem,
   const size_t shared_memory_per_block )
{
   Vector x(
      static_cast< Integer >( num_dofs ),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.25 + 0.00001 * static_cast< Real >( i % 8191 );
      } );
   Vector y( static_cast< size_t >( num_dofs ) );
   y = 0.0;

   auto sigma =
      [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim > & X ) -> Real
      {
         Real value = 1.0;
         for ( Integer d = 0; d < Dim; ++d )
         {
            value += static_cast< Real >( d + 1 ) * X[ d ];
         }
         return value;
      };

   auto op = MakeMassFiniteElementOperator< KernelPolicy >(
      fe_space,
      integration_rule,
      sigma );

   const double time_per_apply = TimeOperator( op, x, y );
   PrintMassRow(
      Dim,
      Order,
      Order + 2,
      num_cells,
      num_dofs,
      num_qpts,
      kernel_name,
      layout_name,
      threaded_dimensions,
      target_threads_per_block,
      KernelPolicy::thread_layout_type::GetNumberOfThreads(),
      batch_size,
      block_dims,
      required_shared_mem,
      shared_memory_per_block,
      time_per_apply,
      "ok" );
}

template <
   Integer Dim,
   Integer Order,
   typename KernelPolicy >
void RunMassKernelPolicy(
   const char * kernel_name,
   const char * layout_name,
   const size_t threaded_dimensions,
   const size_t target_threads_per_block,
   const size_t batch_size )
{
   static constexpr Integer num_quad_1d = Order + 2;

   const auto extents = MassBenchmarkExtents< Dim, Order >();
   const GlobalIndex num_cells = Product( extents );
   const GlobalIndex num_dofs =
      num_cells * DofsPerElement< Dim, Order >();
   const GlobalIndex num_qpts =
      num_cells * QuadraturePointsPerElement< Dim, num_quad_1d >();
   const auto block_dims = BlockDimensions< KernelPolicy >( num_cells );

   auto orders = MakeOrders< Dim, Order >();
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto mesh = MakeCartesianBenchmarkMesh< Dim >( extents );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto num_quads = MakeNumQuads< Dim, num_quad_1d >();
   auto integration_rule = MakeIntegrationRule( num_quads );

   constexpr size_t required_shared_mem =
      Max(
         required_shared_memory_v< KernelPolicy, decltype( integration_rule ) >,
         decltype( fe_space )::finite_element_type::GetNumDofs()
      );
   constexpr size_t shared_memory_per_block =
      KernelContext<
         KernelPolicy,
         required_shared_mem >::shared_memory_block_size;

   if constexpr (
      !MassCaseSupported<
         KernelPolicy,
         decltype( fe_space ),
         decltype( integration_rule ) >::value )
   {
      PrintMassRow(
         Dim,
         Order,
         num_quad_1d,
         num_cells,
         num_dofs,
         num_qpts,
         kernel_name,
         layout_name,
         threaded_dimensions,
         target_threads_per_block,
         KernelPolicy::thread_layout_type::GetNumberOfThreads(),
         batch_size,
         block_dims,
         required_shared_mem,
         shared_memory_per_block,
         0.0,
         "skipped-helper-coverage" );
      return;
   }
   else
   {
      if ( !DeviceThreadLimitsFit( block_dims ) )
      {
         PrintMassRow(
            Dim,
            Order,
            num_quad_1d,
            num_cells,
            num_dofs,
            num_qpts,
            kernel_name,
            layout_name,
            threaded_dimensions,
            target_threads_per_block,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            block_dims,
            required_shared_mem,
            shared_memory_per_block,
            0.0,
            "skipped-launch-limit" );
         return;
      }

      if constexpr (
         shared_memory_per_block * sizeof( Real ) >
         static_shared_memory_compile_limit_bytes )
      {
         PrintMassRow(
            Dim,
            Order,
            num_quad_1d,
            num_cells,
            num_dofs,
            num_qpts,
            kernel_name,
            layout_name,
            threaded_dimensions,
            target_threads_per_block,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            block_dims,
            required_shared_mem,
            shared_memory_per_block,
            0.0,
            "skipped-shared-memory" );
         return;
      }
      else if ( !DeviceSharedMemoryFits( shared_memory_per_block ) )
      {
         PrintMassRow(
            Dim,
            Order,
            num_quad_1d,
            num_cells,
            num_dofs,
            num_qpts,
            kernel_name,
            layout_name,
            threaded_dimensions,
            target_threads_per_block,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            block_dims,
            required_shared_mem,
            shared_memory_per_block,
            0.0,
            "skipped-shared-memory" );
         return;
      }
      else
      {
         RunTimedMassRow<
            Dim,
            Order,
            KernelPolicy >(
            fe_space,
            integration_rule,
            kernel_name,
            layout_name,
            threaded_dimensions,
            target_threads_per_block,
            batch_size,
            num_cells,
            num_dofs,
            num_qpts,
            block_dims,
            required_shared_mem,
            shared_memory_per_block );
      }
   }
}

template <
   Integer Dim,
   Integer Order >
void RunSerialMassCase()
{
   RunMassKernelPolicy<
      Dim,
      Order,
      SerialKernelConfiguration >(
         "SerialKernelConfiguration",
         "ThreadBlockLayout<>",
         0,
         1,
         1 );
}

#if defined( GENDIL_USE_DEVICE )
template < typename ThreadLayout, size_t TargetThreads >
constexpr size_t MassBatchSizeForTarget()
{
   constexpr size_t threads_per_work_item =
      ThreadLayout::GetNumberOfThreads();
   constexpr size_t quotient =
      TargetThreads / threads_per_work_item;
   return quotient > 0 ? quotient : 1;
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunLegacyMassCase(
   const char * layout_name,
   const size_t threaded_dimensions )
{
   using KernelPolicy =
      ThreadFirstKernelConfiguration< ThreadLayout, Dim >;
   constexpr size_t threads_per_work_item =
      ThreadLayout::GetNumberOfThreads();

   RunMassKernelPolicy<
      Dim,
      Order,
      KernelPolicy >(
         "ThreadFirstKernelConfiguration",
         layout_name,
         threaded_dimensions,
         threads_per_work_item,
         1 );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   size_t BatchSize,
   size_t TargetThreads >
void RunDeviceMassCase( const char * layout_name )
{
   using KernelPolicy =
      DeviceKernelConfiguration< ThreadLayout, Dim, BatchSize >;

   RunMassKernelPolicy<
      Dim,
      Order,
      KernelPolicy >(
         "DeviceKernelConfiguration",
         layout_name,
         ThreadLayout::thread_block_dim,
         TargetThreads,
         BatchSize );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   size_t TargetThreads >
void RunDeviceMassTarget( const char * layout_name )
{
   RunDeviceMassCase<
      Dim,
      Order,
      ThreadLayout,
      MassBatchSizeForTarget< ThreadLayout, TargetThreads >(),
      TargetThreads >( layout_name );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunDeviceMassTargetSweep( const char * layout_name )
{
#if defined( GENDIL_USE_CUDA )
   RunDeviceMassTarget< Dim, Order, ThreadLayout, 32 >( layout_name );
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 64 >() !=
      MassBatchSizeForTarget< ThreadLayout, 32 >() )
   {
      RunDeviceMassTarget< Dim, Order, ThreadLayout, 64 >( layout_name );
   }
#else
   RunDeviceMassTarget< Dim, Order, ThreadLayout, 64 >( layout_name );
#endif

   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 128 >() !=
      MassBatchSizeForTarget< ThreadLayout, 64 >() )
   {
      RunDeviceMassTarget< Dim, Order, ThreadLayout, 128 >( layout_name );
   }
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 256 >() !=
      MassBatchSizeForTarget< ThreadLayout, 128 >() )
   {
      RunDeviceMassTarget< Dim, Order, ThreadLayout, 256 >( layout_name );
   }
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 512 >() !=
      MassBatchSizeForTarget< ThreadLayout, 256 >() )
   {
      RunDeviceMassTarget< Dim, Order, ThreadLayout, 512 >( layout_name );
   }
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunMassLayout(
   const char * layout_name,
   const size_t threaded_dimensions )
{
   RunLegacyMassCase< Dim, Order, ThreadLayout >(
      layout_name,
      threaded_dimensions );
   RunDeviceMassTargetSweep< Dim, Order, ThreadLayout >( layout_name );
}
#endif

template < Integer Dim, Integer Order >
void RunMassTensorProductLayouts()
{
#if defined( GENDIL_USE_DEVICE )
   RunMassLayout< Dim, Order, ThreadBlockLayout<> >(
      "ThreadBlockLayout<>",
      0 );
   RunMassLayout< Dim, Order, ThreadBlockLayout< Order + 2 > >(
      "ThreadBlockLayout<num_quad_1d>",
      1 );

   if constexpr ( Dim >= 2 )
   {
      RunMassLayout< Dim, Order, ThreadBlockLayout< Order + 2, Order + 2 > >(
         "ThreadBlockLayout<num_quad_1d,num_quad_1d>",
         2 );
   }

   if constexpr ( Dim >= 3 )
   {
      RunMassLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2, Order + 2 > >(
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>",
            3 );
   }
#endif
}

template < Integer Dim, Integer Order >
void RunMassOrder()
{
   RunSerialMassCase< Dim, Order >();
   RunMassTensorProductLayouts< Dim, Order >();
}

template < Integer Dim >
void RunMassDimension()
{
   RunMassOrder< Dim, 1 >();
   RunMassOrder< Dim, 3 >();
   RunMassOrder< Dim, 5 >();
}

} // namespace gendil::benchmarks
