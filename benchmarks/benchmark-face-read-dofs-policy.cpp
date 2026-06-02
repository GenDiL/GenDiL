// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <type_traits>

using namespace gendil;

namespace
{
#if defined( GENDIL_USE_DEVICE )
constexpr Integer warmup_iterations = 2;
constexpr Integer timed_iterations = 5;
constexpr GlobalIndex target_dofs = 1'000'000;
constexpr GlobalIndex min_cells = 256;
#else
constexpr Integer warmup_iterations = 1;
constexpr Integer timed_iterations = 1;
constexpr GlobalIndex target_dofs = 50'000;
constexpr GlobalIndex min_cells = 64;
#endif
constexpr size_t static_shared_memory_compile_limit_bytes = 64 * 1024;
constexpr Real correctness_tolerance = 1e-10;

template < typename BaseKernelPolicy, typename FaceReadPolicy >
struct KernelPolicyWithFaceReadPolicy : public BaseKernelPolicy
{
   using face_read_dofs_policy = FaceReadPolicy;
};

template < typename KernelPolicy >
struct BatchOneReferenceKernel
{
   using type = KernelPolicy;
};

template <
   typename ThreadLayout,
   size_t MaxSharedDimensions,
   size_t BatchSize >
struct BatchOneReferenceKernel<
   DeviceKernelConfiguration<
      ThreadLayout,
      MaxSharedDimensions,
      BatchSize > >
{
   using type =
      DeviceKernelConfiguration< ThreadLayout, MaxSharedDimensions, 1 >;
};

template < typename KernelPolicy >
using batch_one_reference_kernel_t =
   typename BatchOneReferenceKernel< KernelPolicy >::type;

template < typename Policy >
constexpr const char * FaceReadPolicyName()
{
   if constexpr ( std::is_same_v< Policy, DirectGlobalFaceReadDofsPolicy > )
   {
      return "DirectGlobalFaceReadDofsPolicy";
   }
   else
   {
      return "FullSharedFaceReadDofsPolicy";
   }
}

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
   else if constexpr ( Dim == 3 )
   {
      return Cartesian3DMesh( h, n[ 0 ], n[ 1 ], n[ 2 ] );
   }
   else if constexpr ( Dim == 4 )
   {
      Cartesian2DMesh mesh_1( h, n[ 0 ], n[ 1 ] );
      Cartesian2DMesh mesh_2( h, n[ 2 ], n[ 3 ] );
      return MakeCartesianProductMesh( mesh_1, mesh_2 );
   }
   else if constexpr ( Dim == 5 )
   {
      Cartesian3DMesh mesh_1( h, n[ 0 ], n[ 1 ], n[ 2 ] );
      Cartesian2DMesh mesh_2( h, n[ 3 ], n[ 4 ] );
      return MakeCartesianProductMesh( mesh_1, mesh_2 );
   }
   else
   {
      static_assert( Dim == 6 );
      Cartesian3DMesh mesh_1( h, n[ 0 ], n[ 1 ], n[ 2 ] );
      Cartesian3DMesh mesh_2( h, n[ 3 ], n[ 4 ], n[ 5 ] );
      return MakeCartesianProductMesh( mesh_1, mesh_2 );
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

void PrintFaceHeader()
{
   std::cout
      << "benchmark,dimension,order,num_quad_1d,num_cells,num_faces,"
      << "layout,threads_per_work_item,batch_size,total_threads_per_block,"
      << "face_read_policy,shared_memory_per_work_item,"
      << "shared_memory_per_block,time_per_apply,faces_per_s,"
      << "dofs_read_per_s,status\n";
}

void PrintFaceRow(
   const Integer dim,
   const Integer order,
   const Integer num_quad_1d,
   const GlobalIndex num_cells,
   const GlobalIndex num_faces,
   const GlobalIndex dofs_read_per_apply,
   const char * layout,
   const size_t threads_per_work_item,
   const size_t batch_size,
   const char * face_read_policy,
   const size_t shared_memory_per_work_item,
   const size_t shared_memory_per_block,
   const double time_per_apply,
   const char * status )
{
   const size_t total_threads_per_block =
      threads_per_work_item * batch_size;
   const double faces_per_s =
      time_per_apply > 0.0 ? static_cast< double >( num_faces ) / time_per_apply : 0.0;
   const double dofs_read_per_s =
      time_per_apply > 0.0
         ? static_cast< double >( dofs_read_per_apply ) / time_per_apply
         : 0.0;

   std::cout
      << "face-read-dofs-policy,"
      << dim << ','
      << order << ','
      << num_quad_1d << ','
      << num_cells << ','
      << num_faces << ','
      << layout << ','
      << threads_per_work_item << ','
      << batch_size << ','
      << total_threads_per_block << ','
      << face_read_policy << ','
      << shared_memory_per_work_item << ','
      << shared_memory_per_block << ','
      << time_per_apply << ','
      << faces_per_s << ','
      << dofs_read_per_s << ','
      << status << '\n';
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace >
struct FaceReadCaseSupported
{
   using dof_shape =
      orders_to_num_dofs<
         typename FiniteElementSpace::finite_element_type::
            shape_functions::orders >;

   static constexpr bool value =
      threaded_shape_covered_v< KernelPolicy, dof_shape >;
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

bool VectorsClose( const Vector & a, const Vector & b )
{
   if ( a.Size() != b.Size() )
   {
      return false;
   }

   const Real * a_data = a.ReadHostData();
   const Real * b_data = b.ReadHostData();
   Real max_diff = 0.0;
   Real max_ref = 0.0;
   for ( size_t i = 0; i < a.Size(); ++i )
   {
      max_diff = std::max( max_diff, std::abs( a_data[ i ] - b_data[ i ] ) );
      max_ref = std::max( max_ref, std::abs( a_data[ i ] ) );
   }

   return max_diff <= correctness_tolerance * ( 1.0 + max_ref );
}

template <
   Integer Dim,
   Integer Order,
   typename BaseKernelPolicy,
   typename FaceReadPolicy >
void RunFaceKernelPolicy(
   const char * layout_name,
   const size_t batch_size )
{
   static constexpr Integer num_quad_1d = Order + 2;
   using KernelPolicy =
      KernelPolicyWithFaceReadPolicy< BaseKernelPolicy, FaceReadPolicy >;

   const auto extents = BenchmarkExtents< Dim, Order >();
   const GlobalIndex num_cells = Product( extents );
   const GlobalIndex num_faces =
      num_cells * static_cast< GlobalIndex >( 2 * Dim );
   const GlobalIndex dofs_read_per_apply =
      num_faces * DofsPerElement< Dim, Order >();

   auto orders = MakeOrders< Dim, Order >();
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto mesh = MakeBenchmarkMesh< Dim >( extents );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto num_quads = MakeNumQuads< Dim, num_quad_1d >();
   auto integration_rule = MakeIntegrationRule( num_quads );

   constexpr size_t required_shared_mem =
      face_speed_of_light_required_shared_memory_v<
         FaceSoLType::ReadCell,
         KernelPolicy,
         decltype( fe_space ) >;
   constexpr size_t shared_memory_per_block =
      KernelContext<
         KernelPolicy,
         required_shared_mem >::shared_memory_block_size;

   if constexpr (
      !FaceReadCaseSupported<
         KernelPolicy,
         decltype( fe_space ) >::value )
   {
      PrintFaceRow(
         Dim,
         Order,
         num_quad_1d,
         num_cells,
         num_faces,
         dofs_read_per_apply,
         layout_name,
         KernelPolicy::thread_layout_type::GetNumberOfThreads(),
         batch_size,
         FaceReadPolicyName< FaceReadPolicy >(),
         required_shared_mem,
         shared_memory_per_block,
         0.0,
         "skipped-helper-coverage" );
      return;
   }
   else
   {
      const auto block_dims = BlockDimensions< KernelPolicy >( num_cells );
      if ( !DeviceThreadLimitsFit( block_dims ) )
      {
         PrintFaceRow(
            Dim,
            Order,
            num_quad_1d,
            num_cells,
            num_faces,
            dofs_read_per_apply,
            layout_name,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            FaceReadPolicyName< FaceReadPolicy >(),
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
         PrintFaceRow(
            Dim,
            Order,
            num_quad_1d,
            num_cells,
            num_faces,
            dofs_read_per_apply,
            layout_name,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            FaceReadPolicyName< FaceReadPolicy >(),
            required_shared_mem,
            shared_memory_per_block,
            0.0,
            "skipped-shared-memory" );
         return;
      }
      else
      {
         if ( !DeviceSharedMemoryFits( shared_memory_per_block ) )
         {
            PrintFaceRow(
               Dim,
               Order,
               num_quad_1d,
               num_cells,
               num_faces,
               dofs_read_per_apply,
               layout_name,
               KernelPolicy::thread_layout_type::GetNumberOfThreads(),
               batch_size,
               FaceReadPolicyName< FaceReadPolicy >(),
               required_shared_mem,
               shared_memory_per_block,
               0.0,
               "skipped-shared-memory" );
            return;
         }

         Vector x(
            static_cast< Integer >( fe_space.GetNumberOfFiniteElementDofs() ),
            [] GENDIL_HOST_DEVICE ( Integer i )
            {
               return 0.125 + 0.0001 * static_cast< Real >( i % 1543 );
            } );
         Vector y( fe_space.GetNumberOfFiniteElementDofs() );
         y = 0.0;

         auto op =
            MakeFaceSpeedOfLightOperator< KernelPolicy >(
               fe_space,
               integration_rule );

         if constexpr (
            std::is_same_v< FaceReadPolicy, DirectGlobalFaceReadDofsPolicy > )
         {
            using ReferenceBaseKernel =
               batch_one_reference_kernel_t< BaseKernelPolicy >;
            using ReferenceKernelPolicy =
               KernelPolicyWithFaceReadPolicy<
                  ReferenceBaseKernel,
                  FullSharedFaceReadDofsPolicy >;

            constexpr size_t reference_required_shared_mem =
               face_speed_of_light_required_shared_memory_v<
                  FaceSoLType::ReadCell,
                  ReferenceKernelPolicy,
                  decltype( fe_space ) >;
            constexpr size_t reference_shared_memory_per_block =
               KernelContext<
                  ReferenceKernelPolicy,
                  reference_required_shared_mem >::shared_memory_block_size;

            const auto reference_block_dims =
               BlockDimensions< ReferenceKernelPolicy >( num_cells );
            if constexpr (
               !FaceReadCaseSupported<
                  ReferenceKernelPolicy,
                  decltype( fe_space ) >::value )
            {
               PrintFaceRow(
                  Dim,
                  Order,
                  num_quad_1d,
                  num_cells,
                  num_faces,
                  dofs_read_per_apply,
                  layout_name,
                  KernelPolicy::thread_layout_type::GetNumberOfThreads(),
                  batch_size,
                  FaceReadPolicyName< FaceReadPolicy >(),
                  required_shared_mem,
                  shared_memory_per_block,
                  0.0,
                  "skipped-correctness-reference-helper-coverage" );
               return;
            }

            if constexpr (
               reference_shared_memory_per_block * sizeof( Real ) >
               static_shared_memory_compile_limit_bytes )
            {
               PrintFaceRow(
                  Dim,
                  Order,
                  num_quad_1d,
                  num_cells,
                  num_faces,
                  dofs_read_per_apply,
                  layout_name,
                  KernelPolicy::thread_layout_type::GetNumberOfThreads(),
                  batch_size,
                  FaceReadPolicyName< FaceReadPolicy >(),
                  required_shared_mem,
                  shared_memory_per_block,
                  0.0,
                  "skipped-correctness-reference-shared-memory" );
               return;
            }

            else
            {
               if ( !DeviceThreadLimitsFit( reference_block_dims ) ||
                    !DeviceSharedMemoryFits( reference_shared_memory_per_block ) )
               {
                  PrintFaceRow(
                     Dim,
                     Order,
                     num_quad_1d,
                     num_cells,
                     num_faces,
                     dofs_read_per_apply,
                     layout_name,
                     KernelPolicy::thread_layout_type::GetNumberOfThreads(),
                     batch_size,
                     FaceReadPolicyName< FaceReadPolicy >(),
                     required_shared_mem,
                     shared_memory_per_block,
                     0.0,
                     "skipped-correctness-reference-launch" );
                  return;
               }

               Vector y_reference( fe_space.GetNumberOfFiniteElementDofs() );
               Vector y_direct( fe_space.GetNumberOfFiniteElementDofs() );
               y_reference = 0.0;
               y_direct = 0.0;

               auto reference_op =
                  MakeFaceSpeedOfLightOperator< ReferenceKernelPolicy >(
                     fe_space,
                     integration_rule );
               reference_op( x, y_reference );
               op( x, y_direct );
               GENDIL_DEVICE_SYNC;

               if ( !VectorsClose( y_reference, y_direct ) )
               {
                  PrintFaceRow(
                     Dim,
                     Order,
                     num_quad_1d,
                     num_cells,
                     num_faces,
                     dofs_read_per_apply,
                     layout_name,
                     KernelPolicy::thread_layout_type::GetNumberOfThreads(),
                     batch_size,
                     FaceReadPolicyName< FaceReadPolicy >(),
                     required_shared_mem,
                     shared_memory_per_block,
                     0.0,
                     "skipped-correctness" );
                  return;
               }
            }
      }

      y = 0.0;
      const double time_per_apply = TimeOperator( op, x, y );
      PrintFaceRow(
         Dim,
         Order,
         num_quad_1d,
         num_cells,
         num_faces,
         dofs_read_per_apply,
         layout_name,
         KernelPolicy::thread_layout_type::GetNumberOfThreads(),
         batch_size,
         FaceReadPolicyName< FaceReadPolicy >(),
         required_shared_mem,
         shared_memory_per_block,
         time_per_apply,
         "ok" );
      }
   }
}

template <
   Integer Dim,
   Integer Order >
void RunSerialFaceCases()
{
   RunFaceKernelPolicy<
      Dim,
      Order,
      SerialKernelConfiguration,
      FullSharedFaceReadDofsPolicy >(
         "ThreadBlockLayout<>",
         1 );
   RunFaceKernelPolicy<
      Dim,
      Order,
      SerialKernelConfiguration,
      DirectGlobalFaceReadDofsPolicy >(
         "ThreadBlockLayout<>",
         1 );
}

#if defined( GENDIL_USE_DEVICE )
template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   typename FaceReadPolicy,
   size_t BatchSize >
void RunDeviceFaceCase( const char * layout_name )
{
   using BaseKernelPolicy =
      DeviceKernelConfiguration< ThreadLayout, Dim, BatchSize >;

   RunFaceKernelPolicy<
      Dim,
      Order,
      BaseKernelPolicy,
      FaceReadPolicy >(
         layout_name,
         BatchSize );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   typename FaceReadPolicy >
void RunDeviceFaceBatchSweepForPolicy( const char * layout_name )
{
   RunDeviceFaceCase< Dim, Order, ThreadLayout, FaceReadPolicy, 1 >(
      layout_name );
   RunDeviceFaceCase< Dim, Order, ThreadLayout, FaceReadPolicy, 2 >(
      layout_name );
   RunDeviceFaceCase< Dim, Order, ThreadLayout, FaceReadPolicy, 4 >(
      layout_name );
   RunDeviceFaceCase< Dim, Order, ThreadLayout, FaceReadPolicy, 8 >(
      layout_name );
   RunDeviceFaceCase< Dim, Order, ThreadLayout, FaceReadPolicy, 16 >(
      layout_name );
   if constexpr ( device_warp_size != 32 )
   {
      RunDeviceFaceCase< Dim, Order, ThreadLayout, FaceReadPolicy, 32 >(
         layout_name );
   }
   RunDeviceFaceCase<
      Dim,
      Order,
      ThreadLayout,
      FaceReadPolicy,
      device_warp_size >( layout_name );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunDeviceFaceLayout( const char * layout_name )
{
   RunDeviceFaceBatchSweepForPolicy<
      Dim,
      Order,
      ThreadLayout,
      FullSharedFaceReadDofsPolicy >( layout_name );
   RunDeviceFaceBatchSweepForPolicy<
      Dim,
      Order,
      ThreadLayout,
      DirectGlobalFaceReadDofsPolicy >( layout_name );
}
#endif

template < Integer Dim, Integer Order >
void RunFaceOrder()
{
   RunSerialFaceCases< Dim, Order >();

#if defined( GENDIL_USE_DEVICE )
   RunDeviceFaceLayout< Dim, Order, ThreadBlockLayout<> >(
      "ThreadBlockLayout<>" );
   RunDeviceFaceLayout< Dim, Order, ThreadBlockLayout< Order + 2 > >(
      "ThreadBlockLayout<num_quad_1d>" );
   RunDeviceFaceLayout< Dim, Order, ThreadBlockLayout< 4 > >(
      "ThreadBlockLayout<4>" );
   RunDeviceFaceLayout< Dim, Order, ThreadBlockLayout< 5 > >(
      "ThreadBlockLayout<5>" );
   RunDeviceFaceLayout< Dim, Order, ThreadBlockLayout< 8 > >(
      "ThreadBlockLayout<8>" );

   if constexpr ( Dim >= 2 )
   {
      RunDeviceFaceLayout< Dim, Order, ThreadBlockLayout< 2, 2 > >(
         "ThreadBlockLayout<2,2>" );
      RunDeviceFaceLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2 > >(
            "ThreadBlockLayout<num_quad_1d,num_quad_1d>" );
   }

   if constexpr ( Dim >= 3 )
   {
      RunDeviceFaceLayout< Dim, Order, ThreadBlockLayout< 2, 2, 2 > >(
         "ThreadBlockLayout<2,2,2>" );
      RunDeviceFaceLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2, Order + 2 > >(
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>" );
   }
#endif
}

template < Integer Dim >
void RunFaceDimension()
{
   RunFaceOrder< Dim, 1 >();
   RunFaceOrder< Dim, 3 >();
}
} // namespace

int main()
{
   PrintFaceHeader();
   RunFaceDimension< 1 >();
   RunFaceDimension< 2 >();
   RunFaceDimension< 3 >();
   RunFaceDimension< 4 >();
   RunFaceDimension< 5 >();
   RunFaceDimension< 6 >();
   return 0;
}
