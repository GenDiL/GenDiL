// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <type_traits>

using namespace gendil;

namespace
{
#if defined( GENDIL_USE_DEVICE )
constexpr Integer warmup_iterations = 2;
constexpr Integer timed_iterations = 5;
constexpr GlobalIndex target_dofs = 2'000'000;
constexpr GlobalIndex min_cells = 4'096;
#else
constexpr Integer warmup_iterations = 1;
constexpr Integer timed_iterations = 1;
constexpr GlobalIndex target_dofs = 100'000;
constexpr GlobalIndex min_cells = 512;
#endif
constexpr size_t static_shared_memory_compile_limit_bytes = 96 * 1024;

template < Integer Value, size_t... Is >
auto MakeOrdersImpl( std::index_sequence< Is... > )
{
   return FiniteElementOrders<
      ( static_cast< void >( Is ), Value )... >{};
}

template < Integer Value, size_t... Is >
auto MakeNumQuadsImpl( std::index_sequence< Is... > )
{
   return IntegrationRuleNumPoints<
      ( static_cast< void >( Is ), Value )... >{};
}

template < Integer Dim, Integer Order >
auto MakeOrders()
{
   return MakeOrdersImpl< Order >( std::make_index_sequence< Dim >{} );
}

template < Integer Dim, Integer NumQuad1D >
auto MakeNumQuads()
{
   return MakeNumQuadsImpl< NumQuad1D >(
      std::make_index_sequence< Dim >{} );
}

template < Integer Dim, Integer Order >
constexpr GlobalIndex DofsPerElement()
{
   GlobalIndex value = 1;
   for ( Integer d = 0; d < Dim; ++d )
   {
      value *= static_cast< GlobalIndex >( Order + 1 );
   }
   return value;
}

template < Integer Dim, Integer NumQuad1D >
constexpr GlobalIndex QuadraturePointsPerElement()
{
   GlobalIndex value = 1;
   for ( Integer d = 0; d < Dim; ++d )
   {
      value *= static_cast< GlobalIndex >( NumQuad1D );
   }
   return value;
}

template < Integer Dim >
std::array< Integer, Dim > BalancedExtents( GlobalIndex target_cells )
{
   std::array< Integer, Dim > extents{};
   extents.fill( 1 );

   GlobalIndex product = 1;
   Integer axis = 0;
   while ( product < target_cells )
   {
      product /= static_cast< GlobalIndex >( extents[ axis ] );
      extents[ axis ] *= 2;
      product *= static_cast< GlobalIndex >( extents[ axis ] );
      axis = ( axis + 1 ) % Dim;
   }

   return extents;
}

template < Integer Dim, Integer Order >
std::array< Integer, Dim > BenchmarkExtents()
{
   const GlobalIndex cells_from_dofs =
      target_dofs / DofsPerElement< Dim, Order >();
   const GlobalIndex target_cells =
      cells_from_dofs > min_cells ? cells_from_dofs : min_cells;
   return BalancedExtents< Dim >( target_cells );
}

template < Integer Dim >
GlobalIndex Product( const std::array< Integer, Dim > & extents )
{
   GlobalIndex product = 1;
   for ( Integer d = 0; d < Dim; ++d )
   {
      product *= static_cast< GlobalIndex >( extents[ d ] );
   }
   return product;
}

template < Integer Dim >
auto MakeBenchmarkMesh( const std::array< Integer, Dim > & n )
{
   constexpr Real h = 1.0;
   if constexpr ( Dim == 1 )
   {
      return Cartesian1DMesh( h, n[ 0 ] );
   }
   else if constexpr ( Dim == 2 )
   {
      return Cartesian2DMesh( h, n[ 0 ], n[ 1 ] );
   }
   else
   {
      static_assert( Dim == 3 );
      return Cartesian3DMesh( h, n[ 0 ], n[ 1 ], n[ 2 ] );
   }
}

template < typename KernelPolicy >
std::array< GlobalIndex, 3 > BlockDimensions( const GlobalIndex num_items )
{
   if constexpr ( requires { KernelPolicy::GetLaunchGeometry( num_items ); } )
   {
      const auto geometry = KernelPolicy::GetLaunchGeometry( num_items );
      return { geometry.block_x, geometry.block_y, geometry.block_z };
   }
   else
   {
      return { 1, 1, 1 };
   }
}

bool DeviceThreadLimitsFit(
   const std::array< GlobalIndex, 3 > & block_dims )
{
#if defined( GENDIL_USE_DEVICE )
   int device = 0;
   GENDIL_DEVICE_CHECK( DeviceGetDevice( &device ) );

   DeviceProperties props;
   GENDIL_DEVICE_CHECK( DeviceGetDeviceProperties( &props, device ) );

   const auto threads_per_block =
      block_dims[ 0 ] * block_dims[ 1 ] * block_dims[ 2 ];
   return block_dims[ 0 ] <= static_cast< GlobalIndex >( props.maxThreadsDim[ 0 ] ) &&
          block_dims[ 1 ] <= static_cast< GlobalIndex >( props.maxThreadsDim[ 1 ] ) &&
          block_dims[ 2 ] <= static_cast< GlobalIndex >( props.maxThreadsDim[ 2 ] ) &&
          threads_per_block <= static_cast< GlobalIndex >( props.maxThreadsPerBlock );
#else
   (void) block_dims;
   return true;
#endif
}

bool DeviceSharedMemoryFits( const size_t shared_memory_per_block )
{
#if defined( GENDIL_USE_DEVICE )
   int device = 0;
   GENDIL_DEVICE_CHECK( DeviceGetDevice( &device ) );

   DeviceProperties props;
   GENDIL_DEVICE_CHECK( DeviceGetDeviceProperties( &props, device ) );

   return shared_memory_per_block * sizeof( Real ) <=
      static_cast< size_t >( props.sharedMemPerBlock );
#else
   (void) shared_memory_per_block;
   return true;
#endif
}

void PrintMassHeader()
{
   std::cout
      << "benchmark,dimension,order,num_quad_1d,num_cells,num_dofs,"
      << "num_qpts,kernel,layout,threads_per_work_item,batch_size,"
      << "total_threads_per_block,block_x,block_y,block_z,"
      << "shared_memory_per_work_item,shared_memory_per_block,"
      << "time_per_apply,elements_per_s,dofs_per_s,qpts_per_s,status\n";
}

void PrintMassRow(
   const Integer dim,
   const Integer order,
   const Integer num_quad_1d,
   const GlobalIndex num_cells,
   const GlobalIndex num_dofs,
   const GlobalIndex num_qpts,
   const char * kernel,
   const char * layout,
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
      time_per_apply > 0.0 ? static_cast< double >( num_cells ) / time_per_apply : 0.0;
   const double dofs_per_s =
      time_per_apply > 0.0 ? static_cast< double >( num_dofs ) / time_per_apply : 0.0;
   const double qpts_per_s =
      time_per_apply > 0.0 ? static_cast< double >( num_qpts ) / time_per_apply : 0.0;

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

template < typename Operator >
double TimeOperator( Operator & op, Vector & x, Vector & y )
{
   for ( Integer iter = 0; iter < warmup_iterations; ++iter )
   {
      op( x, y );
      op( y, x );
   }
   GENDIL_DEVICE_SYNC;

   const auto start = std::chrono::steady_clock::now();
   for ( Integer iter = 0; iter < timed_iterations; ++iter )
   {
      op( x, y );
      op( y, x );
   }
   GENDIL_DEVICE_SYNC;
   const auto end = std::chrono::steady_clock::now();

   return std::chrono::duration< double >( end - start ).count() /
      static_cast< double >( 2 * timed_iterations );
}

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
   const size_t batch_size )
{
   static constexpr Integer num_quad_1d = Order + 2;

   const auto extents = BenchmarkExtents< Dim, Order >();
   const GlobalIndex num_cells = Product( extents );
   const GlobalIndex num_dofs =
      num_cells * DofsPerElement< Dim, Order >();
   const GlobalIndex num_qpts =
      num_cells * QuadraturePointsPerElement< Dim, num_quad_1d >();
   const auto block_dims = BlockDimensions< KernelPolicy >( num_cells );

   auto orders = MakeOrders< Dim, Order >();
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto mesh = MakeBenchmarkMesh< Dim >( extents );
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
         1 );
}

#if defined( GENDIL_USE_DEVICE )
template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunLegacyMassCase( const char * layout_name )
{
   using KernelPolicy =
      ThreadFirstKernelConfiguration< ThreadLayout, Dim >;

   RunMassKernelPolicy<
      Dim,
      Order,
      KernelPolicy >(
         "ThreadFirstKernelConfiguration",
         layout_name,
         1 );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   size_t BatchSize >
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
         BatchSize );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunDeviceMassBatchSweep( const char * layout_name )
{
   RunDeviceMassCase< Dim, Order, ThreadLayout, 1 >( layout_name );
   RunDeviceMassCase< Dim, Order, ThreadLayout, 2 >( layout_name );
   RunDeviceMassCase< Dim, Order, ThreadLayout, 4 >( layout_name );
   RunDeviceMassCase< Dim, Order, ThreadLayout, 8 >( layout_name );
   RunDeviceMassCase< Dim, Order, ThreadLayout, 16 >( layout_name );
   if constexpr ( device_warp_size != 32 )
   {
      RunDeviceMassCase< Dim, Order, ThreadLayout, 32 >( layout_name );
   }
   RunDeviceMassCase< Dim, Order, ThreadLayout, device_warp_size >(
      layout_name );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunMassLayout( const char * layout_name )
{
   RunLegacyMassCase< Dim, Order, ThreadLayout >( layout_name );
   RunDeviceMassBatchSweep< Dim, Order, ThreadLayout >( layout_name );
}
#endif

template < Integer Dim, Integer Order >
void RunMassOrder()
{
   RunSerialMassCase< Dim, Order >();

#if defined( GENDIL_USE_DEVICE )
   RunMassLayout< Dim, Order, ThreadBlockLayout<> >(
      "ThreadBlockLayout<>" );
   RunMassLayout< Dim, Order, ThreadBlockLayout< Order + 2 > >(
      "ThreadBlockLayout<num_quad_1d>" );
   RunMassLayout< Dim, Order, ThreadBlockLayout< 4 > >(
      "ThreadBlockLayout<4>" );
   RunMassLayout< Dim, Order, ThreadBlockLayout< 5 > >(
      "ThreadBlockLayout<5>" );
   RunMassLayout< Dim, Order, ThreadBlockLayout< 8 > >(
      "ThreadBlockLayout<8>" );

   if constexpr ( Dim >= 2 )
   {
      RunMassLayout< Dim, Order, ThreadBlockLayout< 2, 2 > >(
         "ThreadBlockLayout<2,2>" );
      RunMassLayout< Dim, Order, ThreadBlockLayout< Order + 2, Order + 2 > >(
         "ThreadBlockLayout<num_quad_1d,num_quad_1d>" );
   }

   if constexpr ( Dim >= 3 )
   {
      RunMassLayout< Dim, Order, ThreadBlockLayout< 2, 2, 2 > >(
         "ThreadBlockLayout<2,2,2>" );
      RunMassLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2, Order + 2 > >(
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>" );
   }
#endif
}

template < Integer Dim >
void RunMassDimension()
{
   RunMassOrder< Dim, 1 >();
   RunMassOrder< Dim, 3 >();
   RunMassOrder< Dim, 5 >();
}
} // namespace

int main()
{
   PrintMassHeader();
   RunMassDimension< 1 >();
   RunMassDimension< 2 >();
   RunMassDimension< 3 >();
   return 0;
}
