// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "benchmark-common.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

namespace gendil::benchmarks
{

constexpr GlobalIndex mass_smoke_target_dofs = 1 * 1024 * 1024;
constexpr GlobalIndex mass_performance_target_dofs = 32 * 1024 * 1024;

struct MassBenchmarkOptions
{
   GlobalIndex target_num_dofs = mass_performance_target_dofs;
};

enum class MassBenchmarkOptionParseResult
{
   run,
   exit_success,
   exit_failure
};

inline void PrintMassUsage( const char * executable, std::ostream & os )
{
   os << "Usage: " << executable
      << " [--performance] [--smoke] [--target-dofs N]\n"
      << "  --performance    use the default 32M DoF target\n"
      << "  --smoke          use a quick 1M DoF target\n"
      << "  --target-dofs N  use an explicit positive DoF target\n";
}

inline bool MassParsePositiveGlobalIndex(
   const char * value,
   GlobalIndex & parsed )
{
   if ( value == nullptr || value[ 0 ] == '\0' )
   {
      return false;
   }

   errno = 0;
   char * end = nullptr;
   const unsigned long long raw = std::strtoull( value, &end, 10 );
   if ( errno != 0 || end == value || *end != '\0' || raw == 0 )
   {
      return false;
   }
   if ( raw > std::numeric_limits< GlobalIndex >::max() )
   {
      return false;
   }

   parsed = static_cast< GlobalIndex >( raw );
   return true;
}

inline MassBenchmarkOptionParseResult ParseMassBenchmarkOptions(
   const int argc,
   char ** argv,
   MassBenchmarkOptions & options,
   std::ostream & os )
{
   for ( int i = 1; i < argc; ++i )
   {
      if ( std::strcmp( argv[ i ], "--performance" ) == 0 )
      {
         options.target_num_dofs = mass_performance_target_dofs;
      }
      else if ( std::strcmp( argv[ i ], "--smoke" ) == 0 )
      {
         options.target_num_dofs = mass_smoke_target_dofs;
      }
      else if ( std::strcmp( argv[ i ], "--target-dofs" ) == 0 )
      {
         if ( i + 1 >= argc ||
              !MassParsePositiveGlobalIndex(
                 argv[ i + 1 ],
                 options.target_num_dofs ) )
         {
            PrintMassUsage( argv[ 0 ], os );
            return MassBenchmarkOptionParseResult::exit_failure;
         }
         ++i;
      }
      else if ( std::strcmp( argv[ i ], "--help" ) == 0 ||
                std::strcmp( argv[ i ], "-h" ) == 0 )
      {
         PrintMassUsage( argv[ 0 ], os );
         return MassBenchmarkOptionParseResult::exit_success;
      }
      else
      {
         PrintMassUsage( argv[ 0 ], os );
         return MassBenchmarkOptionParseResult::exit_failure;
      }
   }

   return MassBenchmarkOptionParseResult::run;
}

template < Integer Dim, Integer Order >
std::array< Integer, Dim > MassBenchmarkExtents(
   const GlobalIndex target_num_dofs )
{
   const GlobalIndex dofs_per_cell = DofsPerElement< Dim, Order >();
   const GlobalIndex target_cells =
      BenchmarkClampCellExtent(
         BenchmarkCeilDivideGlobalIndex( target_num_dofs, dofs_per_cell ) );
   return ToIntegerExtents( BalancedBenchmarkExtents< Dim >( target_cells ) );
}

template < Integer Dim >
std::array< GlobalIndex, 6 > MassExtentsTo6(
   const std::array< Integer, Dim > & extents )
{
   std::array< GlobalIndex, 6 > padded{ 1, 1, 1, 1, 1, 1 };
   for ( Integer d = 0; d < Dim; ++d )
   {
      padded[ d ] = static_cast< GlobalIndex >( extents[ d ] );
   }
   return padded;
}

inline void PrintMassHeader()
{
   std::cout
      << "benchmark,dimension,order,num_quad_1d,num_cells,num_dofs,"
      << "num_qpts,target_num_dofs,actual_num_dofs,"
      << "extent_0,extent_1,extent_2,extent_3,extent_4,extent_5,"
      << "kernel,layout,threaded_dimensions,"
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
   const GlobalIndex target_num_dofs,
   const GlobalIndex actual_num_dofs,
   const std::array< GlobalIndex, 6 > & extents,
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
      << target_num_dofs << ','
      << actual_num_dofs << ','
      << extents[ 0 ] << ','
      << extents[ 1 ] << ','
      << extents[ 2 ] << ','
      << extents[ 3 ] << ','
      << extents[ 4 ] << ','
      << extents[ 5 ] << ','
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
   const GlobalIndex target_num_dofs,
   const std::array< GlobalIndex, 6 > & extents,
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
      target_num_dofs,
      num_dofs,
      extents,
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
   const size_t batch_size,
   const GlobalIndex target_num_dofs )
{
   static constexpr Integer num_quad_1d = Order + 2;

   const auto extents =
      MassBenchmarkExtents< Dim, Order >( target_num_dofs );
   const auto padded_extents = MassExtentsTo6( extents );
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
         target_num_dofs,
         num_dofs,
         padded_extents,
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
            target_num_dofs,
            num_dofs,
            padded_extents,
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
            target_num_dofs,
            num_dofs,
            padded_extents,
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
            target_num_dofs,
            num_dofs,
            padded_extents,
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
            target_num_dofs,
            padded_extents,
            block_dims,
            required_shared_mem,
            shared_memory_per_block );
      }
   }
}

template <
   Integer Dim,
   Integer Order >
void RunSerialMassCase( const GlobalIndex target_num_dofs )
{
   RunMassKernelPolicy<
      Dim,
      Order,
      SerialKernelConfiguration >(
         "SerialKernelConfiguration",
         "ThreadBlockLayout<>",
         0,
         1,
         1,
         target_num_dofs );
}

#if defined( GENDIL_USE_DEVICE )
template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   size_t BatchSize >
void PrintSkippedMassRow(
   const GlobalIndex target_num_dofs,
   const char * kernel_name,
   const char * layout_name,
   const size_t threaded_dimensions,
   const size_t target_threads_per_block )
{
   static constexpr Integer num_quad_1d = Order + 2;

   const auto extents =
      MassBenchmarkExtents< Dim, Order >( target_num_dofs );
   const auto padded_extents = MassExtentsTo6( extents );
   const GlobalIndex num_cells = Product( extents );
   const GlobalIndex num_dofs =
      num_cells * DofsPerElement< Dim, Order >();
   const GlobalIndex num_qpts =
      num_cells * QuadraturePointsPerElement< Dim, num_quad_1d >();
   constexpr size_t shared_memory_per_work_item =
      Max(
         static_cast< size_t >( DofsPerElement< Dim, Order >() ),
         static_cast< size_t >(
            QuadraturePointsPerElement< Dim, num_quad_1d >() )
      );
   const std::array< GlobalIndex, 3 > block_dims{
      ThreadLayout::GetNumberOfThreads(),
      BatchSize,
      1
   };

   PrintMassRow(
      Dim,
      Order,
      num_quad_1d,
      num_cells,
      num_dofs,
      num_qpts,
      target_num_dofs,
      num_dofs,
      padded_extents,
      kernel_name,
      layout_name,
      threaded_dimensions,
      target_threads_per_block,
      ThreadLayout::GetNumberOfThreads(),
      BatchSize,
      block_dims,
      shared_memory_per_work_item,
      shared_memory_per_work_item * BatchSize,
      0.0,
      "skipped-launch-limit" );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void PrintSkippedLegacyMassCase(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions )
{
   PrintSkippedMassRow<
      Dim,
      Order,
      ThreadLayout,
      1 >(
         target_num_dofs,
         "ThreadFirstKernelConfiguration",
         layout_name,
         threaded_dimensions,
         ThreadLayout::GetNumberOfThreads() );
}

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
   typename ThreadLayout,
   size_t TargetThreads >
void PrintSkippedDeviceMassTarget(
   const GlobalIndex target_num_dofs,
   const char * layout_name )
{
   PrintSkippedMassRow<
      Dim,
      Order,
      ThreadLayout,
      MassBatchSizeForTarget< ThreadLayout, TargetThreads >() >(
         target_num_dofs,
         "DeviceKernelConfiguration",
         layout_name,
         ThreadLayout::thread_block_dim,
         TargetThreads );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void PrintSkippedDeviceMassTargetSweep(
   const GlobalIndex target_num_dofs,
   const char * layout_name )
{
#if defined( GENDIL_USE_CUDA )
   PrintSkippedDeviceMassTarget< Dim, Order, ThreadLayout, 32 >(
      target_num_dofs,
      layout_name );
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 64 >() !=
      MassBatchSizeForTarget< ThreadLayout, 32 >() )
   {
      PrintSkippedDeviceMassTarget< Dim, Order, ThreadLayout, 64 >(
         target_num_dofs,
         layout_name );
   }
#else
   PrintSkippedDeviceMassTarget< Dim, Order, ThreadLayout, 64 >(
      target_num_dofs,
      layout_name );
#endif

   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 128 >() !=
      MassBatchSizeForTarget< ThreadLayout, 64 >() )
   {
      PrintSkippedDeviceMassTarget< Dim, Order, ThreadLayout, 128 >(
         target_num_dofs,
         layout_name );
   }
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 256 >() !=
      MassBatchSizeForTarget< ThreadLayout, 128 >() )
   {
      PrintSkippedDeviceMassTarget< Dim, Order, ThreadLayout, 256 >(
         target_num_dofs,
         layout_name );
   }
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 512 >() !=
      MassBatchSizeForTarget< ThreadLayout, 256 >() )
   {
      PrintSkippedDeviceMassTarget< Dim, Order, ThreadLayout, 512 >(
         target_num_dofs,
         layout_name );
   }
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunLegacyMassCase(
   const char * layout_name,
   const size_t threaded_dimensions,
   const GlobalIndex target_num_dofs )
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
         1,
         target_num_dofs );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   size_t BatchSize,
   size_t TargetThreads >
void RunDeviceMassCase(
   const char * layout_name,
   const GlobalIndex target_num_dofs )
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
         BatchSize,
         target_num_dofs );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   size_t TargetThreads >
void RunDeviceMassTarget(
   const char * layout_name,
   const GlobalIndex target_num_dofs )
{
   RunDeviceMassCase<
      Dim,
      Order,
      ThreadLayout,
      MassBatchSizeForTarget< ThreadLayout, TargetThreads >(),
      TargetThreads >(
         layout_name,
         target_num_dofs );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunDeviceMassTargetSweep(
   const char * layout_name,
   const GlobalIndex target_num_dofs )
{
#if defined( GENDIL_USE_CUDA )
   RunDeviceMassTarget< Dim, Order, ThreadLayout, 32 >(
      layout_name,
      target_num_dofs );
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 64 >() !=
      MassBatchSizeForTarget< ThreadLayout, 32 >() )
   {
      RunDeviceMassTarget< Dim, Order, ThreadLayout, 64 >(
         layout_name,
         target_num_dofs );
   }
#else
   RunDeviceMassTarget< Dim, Order, ThreadLayout, 64 >(
      layout_name,
      target_num_dofs );
#endif

   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 128 >() !=
      MassBatchSizeForTarget< ThreadLayout, 64 >() )
   {
      RunDeviceMassTarget< Dim, Order, ThreadLayout, 128 >(
         layout_name,
         target_num_dofs );
   }
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 256 >() !=
      MassBatchSizeForTarget< ThreadLayout, 128 >() )
   {
      RunDeviceMassTarget< Dim, Order, ThreadLayout, 256 >(
         layout_name,
         target_num_dofs );
   }
   if constexpr (
      MassBatchSizeForTarget< ThreadLayout, 512 >() !=
      MassBatchSizeForTarget< ThreadLayout, 256 >() )
   {
      RunDeviceMassTarget< Dim, Order, ThreadLayout, 512 >(
         layout_name,
         target_num_dofs );
   }
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunMassLayout(
   const char * layout_name,
   const size_t threaded_dimensions,
   const GlobalIndex target_num_dofs )
{
   if constexpr (
      ThreadLayout::GetNumberOfThreads() >
      max_threads_per_work_item )
   {
      PrintSkippedLegacyMassCase< Dim, Order, ThreadLayout >(
         target_num_dofs,
         layout_name,
         threaded_dimensions );
      PrintSkippedDeviceMassTargetSweep< Dim, Order, ThreadLayout >(
         target_num_dofs,
         layout_name );
   }
   else
   {
      RunLegacyMassCase< Dim, Order, ThreadLayout >(
         layout_name,
         threaded_dimensions,
         target_num_dofs );
      RunDeviceMassTargetSweep< Dim, Order, ThreadLayout >(
         layout_name,
         target_num_dofs );
   }
}
#endif

template < Integer Dim, Integer Order >
void RunMassTensorProductLayouts( const GlobalIndex target_num_dofs )
{
#if defined( GENDIL_USE_DEVICE )
   RunMassLayout< Dim, Order, ThreadBlockLayout<> >(
      "ThreadBlockLayout<>",
      0,
      target_num_dofs );
   RunMassLayout< Dim, Order, ThreadBlockLayout< Order + 2 > >(
      "ThreadBlockLayout<num_quad_1d>",
      1,
      target_num_dofs );

   if constexpr ( Dim >= 2 )
   {
      RunMassLayout< Dim, Order, ThreadBlockLayout< Order + 2, Order + 2 > >(
         "ThreadBlockLayout<num_quad_1d,num_quad_1d>",
         2,
         target_num_dofs );
   }

   if constexpr ( Dim >= 3 )
   {
      RunMassLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2, Order + 2 > >(
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>",
            3,
            target_num_dofs );
   }

   if constexpr ( Dim >= 4 )
   {
      RunMassLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            4,
            target_num_dofs );
   }

   if constexpr ( Dim >= 5 )
   {
      RunMassLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            5,
            target_num_dofs );
   }

   if constexpr ( Dim >= 6 )
   {
      RunMassLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            6,
            target_num_dofs );
   }
#endif
}

template < Integer Dim, Integer Order >
void RunMassOrder( const GlobalIndex target_num_dofs )
{
   RunSerialMassCase< Dim, Order >( target_num_dofs );
   RunMassTensorProductLayouts< Dim, Order >( target_num_dofs );
}

template < Integer Dim >
void RunMassDimension( const GlobalIndex target_num_dofs )
{
   RunMassOrder< Dim, 0 >( target_num_dofs );
   RunMassOrder< Dim, 1 >( target_num_dofs );
   RunMassOrder< Dim, 2 >( target_num_dofs );
   RunMassOrder< Dim, 3 >( target_num_dofs );
   RunMassOrder< Dim, 5 >( target_num_dofs );
}

} // namespace gendil::benchmarks
