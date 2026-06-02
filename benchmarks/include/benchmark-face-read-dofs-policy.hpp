// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "benchmark-common.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>

namespace gendil::benchmarks
{

#if defined( GENDIL_USE_DEVICE )
constexpr GlobalIndex face_target_dofs = 1'000'000;
constexpr GlobalIndex face_min_cells = 256;
#else
constexpr GlobalIndex face_target_dofs = 50'000;
constexpr GlobalIndex face_min_cells = 64;
#endif
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

template < Integer Dim, Integer Order >
std::array< Integer, Dim > FaceBenchmarkExtents()
{
   const GlobalIndex cells_from_dofs =
      face_target_dofs / DofsPerElement< Dim, Order >();
   const GlobalIndex target_cells =
      cells_from_dofs > face_min_cells ? cells_from_dofs : face_min_cells;
   return BalancedExtents< Dim >( target_cells );
}

inline void PrintFaceHeader()
{
   std::cout
      << "benchmark,dimension,order,num_quad_1d,num_cells,num_faces,"
      << "layout,threads_per_work_item,batch_size,total_threads_per_block,"
      << "face_read_policy,shared_memory_per_work_item,"
      << "shared_memory_per_block,time_per_apply,faces_per_s,"
      << "dofs_read_per_s,status\n";
}

inline void PrintFaceRow(
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
      time_per_apply > 0.0
         ? static_cast< double >( num_faces ) / time_per_apply
         : 0.0;
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

inline bool VectorsClose( const Vector & a, const Vector & b )
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

   const auto extents = FaceBenchmarkExtents< Dim, Order >();
   const GlobalIndex num_cells = Product( extents );
   const GlobalIndex num_faces =
      num_cells * static_cast< GlobalIndex >( 2 * Dim );
   const GlobalIndex dofs_read_per_apply =
      num_faces * DofsPerElement< Dim, Order >();

   auto orders = MakeOrders< Dim, Order >();
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto mesh = MakeCartesianBenchmarkMesh< Dim >( extents );
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

} // namespace gendil::benchmarks
