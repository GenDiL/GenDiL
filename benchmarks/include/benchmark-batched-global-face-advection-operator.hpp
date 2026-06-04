// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "benchmark-common.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

namespace gendil::benchmarks
{

constexpr GlobalIndex global_face_advection_smoke_target_dofs =
   1 * 1024 * 1024;
constexpr GlobalIndex global_face_advection_performance_target_dofs =
   32 * 1024 * 1024;
constexpr Real global_face_advection_correctness_tolerance = 1e-10;

struct GlobalFaceAdvectionBenchmarkOptions
{
   GlobalIndex target_num_dofs =
      global_face_advection_performance_target_dofs;
};

enum class BenchmarkOptionParseResult
{
   run,
   exit_success,
   exit_failure
};

inline void PrintGlobalFaceAdvectionUsage(
   const char * executable,
   std::ostream & os )
{
   os << "Usage: " << executable
      << " [--performance] [--smoke] [--target-dofs N]\n"
      << "  --performance    use the default 32M DoF target\n"
      << "  --smoke          use a quick 1M DoF target\n"
      << "  --target-dofs N  use an explicit positive DoF target\n";
}

inline bool ParsePositiveGlobalIndex(
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

inline BenchmarkOptionParseResult ParseGlobalFaceAdvectionBenchmarkOptions(
   const int argc,
   char ** argv,
   GlobalFaceAdvectionBenchmarkOptions & options,
   std::ostream & os )
{
   for ( int i = 1; i < argc; ++i )
   {
      if ( std::strcmp( argv[ i ], "--performance" ) == 0 )
      {
         options.target_num_dofs =
            global_face_advection_performance_target_dofs;
      }
      else if ( std::strcmp( argv[ i ], "--smoke" ) == 0 )
      {
         options.target_num_dofs =
            global_face_advection_smoke_target_dofs;
      }
      else if ( std::strcmp( argv[ i ], "--target-dofs" ) == 0 )
      {
         if ( i + 1 >= argc ||
              !ParsePositiveGlobalIndex(
                 argv[ i + 1 ],
                 options.target_num_dofs ) )
         {
            PrintGlobalFaceAdvectionUsage( argv[ 0 ], os );
            return BenchmarkOptionParseResult::exit_failure;
         }
         ++i;
      }
      else if ( std::strcmp( argv[ i ], "--help" ) == 0 ||
                std::strcmp( argv[ i ], "-h" ) == 0 )
      {
         PrintGlobalFaceAdvectionUsage( argv[ 0 ], os );
         return BenchmarkOptionParseResult::exit_success;
      }
      else
      {
         PrintGlobalFaceAdvectionUsage( argv[ 0 ], os );
         return BenchmarkOptionParseResult::exit_failure;
      }
   }

   return BenchmarkOptionParseResult::run;
}

template <
   typename BaseKernelPolicy,
   typename FaceReadPolicy,
   typename FaceWritePolicy >
struct KernelPolicyWithFacePolicies : public BaseKernelPolicy
{
   using face_read_dofs_policy = FaceReadPolicy;
   using face_write_dofs_policy = FaceWritePolicy;
};

template < typename BaseKernelPolicy >
using DefaultFaceDofPolicies = BaseKernelPolicy;

template < typename BaseKernelPolicy >
using FullSharedFaceDofPolicies =
   KernelPolicyWithFacePolicies<
      BaseKernelPolicy,
      FullSharedFaceReadDofsPolicy,
      FullSharedFaceWriteDofsPolicy >;

template < Integer Dim, Integer Order >
std::array< GlobalIndex, Dim > GlobalFaceAdvectionBenchmarkExtents(
   const GlobalIndex target_num_dofs )
{
   const GlobalIndex dofs_per_cell = DofsPerElement< Dim, Order >();
   const GlobalIndex target_cells =
      BenchmarkClampCellExtent(
         BenchmarkCeilDivideGlobalIndex( target_num_dofs, dofs_per_cell ) );
   return BalancedBenchmarkExtents< Dim >( target_cells );
}

template < Integer Dim >
auto MakeCartesianBenchmarkMeshFromGlobalExtents(
   const std::array< GlobalIndex, Dim > & extents )
{
   return MakeCartesianBenchmarkMesh< Dim >( ToIntegerExtents( extents ) );
}

template < Integer Dim >
GlobalIndex CountFacesInDirection(
   const std::array< GlobalIndex, Dim > & extents,
   const Integer direction )
{
   if ( direction >= Dim || extents[ direction ] == 0 )
   {
      return 0;
   }

   GlobalIndex num_faces =
      extents[ direction ] > 0 ? extents[ direction ] - 1 : 0;
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( d != direction )
      {
         num_faces *= extents[ d ];
      }
   }
   return num_faces;
}

template < Integer Dim >
std::array< GlobalIndex, 6 > CountFacesByDirection(
   const std::array< GlobalIndex, Dim > & extents )
{
   std::array< GlobalIndex, 6 > face_counts{ 0, 0, 0, 0, 0, 0 };
   for ( Integer d = 0; d < Dim; ++d )
   {
      face_counts[ d ] = CountFacesInDirection( extents, d );
   }
   return face_counts;
}

inline GlobalIndex SumSixColumns(
   const std::array< GlobalIndex, 6 > & values )
{
   GlobalIndex sum = 0;
   for ( Integer d = 0; d < 6; ++d )
   {
      sum += values[ d ];
   }
   return sum;
}

template < typename FaceMeshes >
GlobalIndex CountGlobalFaces( const FaceMeshes & face_meshes )
{
   GlobalIndex num_faces = 0;
   mesh::ForEachFaceMesh(
      face_meshes,
      [&] ( const auto & face_mesh )
      {
         num_faces += face_mesh.GetNumberOfFaces();
      } );
   return num_faces;
}

template < Integer Dim, Integer NumQuad1D >
constexpr GlobalIndex FaceQuadraturePointsPerFace()
{
   GlobalIndex value = 1;
   for ( Integer d = 1; d < Dim; ++d )
   {
      value *= static_cast< GlobalIndex >( NumQuad1D );
   }
   return value;
}

constexpr size_t StaticPower( const size_t base, const size_t exponent )
{
   size_t value = 1;
   for ( size_t i = 0; i < exponent; ++i )
   {
      value *= base;
   }
   return value;
}

template < Integer Dim, Integer Order, bool FullSharedFacePolicy >
constexpr size_t GlobalFaceAdvectionRequiredSharedMemoryEstimate()
{
   constexpr size_t interpolation_and_test_requirement =
      2 * StaticPower( Order + 2, Dim );
   constexpr size_t face_staging_requirement =
      FullSharedFacePolicy ? StaticPower( Order + 1, Dim ) : 0;
   return interpolation_and_test_requirement > face_staging_requirement
      ? interpolation_and_test_requirement
      : face_staging_requirement;
}

inline void PrintGlobalFaceAdvectionHeader()
{
   std::cout
      << "benchmark,dimension,order,num_quad_1d,target_num_dofs,"
      << "actual_num_dofs,num_cells,num_faces_total,num_faces_0,"
      << "num_faces_1,num_faces_2,num_faces_3,num_faces_4,"
      << "num_faces_5,num_dofs,extent_0,extent_1,extent_2,"
      << "extent_3,extent_4,extent_5,"
      << "layout,threaded_dimensions,kernel_family,kernel,face_policy,"
      << "target_threads_per_block,threads_per_work_item,batch_size,"
      << "total_threads_per_block,block_x,block_y,block_z,"
      << "shared_memory_per_work_item,shared_memory_per_block,"
      << "time_per_apply,faces_per_s,dofs_read_per_s,dofs_written_per_s,"
      << "qpts_per_s,status\n";
}

inline void PrintGlobalFaceAdvectionRow(
   const Integer dim,
   const Integer order,
   const Integer num_quad_1d,
   const GlobalIndex target_num_dofs,
   const GlobalIndex actual_num_dofs,
   const GlobalIndex num_cells,
   const std::array< GlobalIndex, 6 > & face_counts,
   const std::array< GlobalIndex, 6 > & extents,
   const char * layout,
   const size_t threaded_dimensions,
   const char * kernel_family,
   const char * kernel,
   const char * face_policy,
   const size_t target_threads_per_block,
   const size_t threads_per_work_item,
   const size_t batch_size,
   const std::array< GlobalIndex, 3 > & block_dims,
   const size_t shared_memory_per_work_item,
   const size_t shared_memory_per_block,
   const double time_per_apply,
   const GlobalIndex dofs_read_per_apply,
   const GlobalIndex dofs_written_per_apply,
   const GlobalIndex qpts_per_apply,
   const char * status )
{
   const GlobalIndex num_faces_total = SumSixColumns( face_counts );
   const size_t total_threads_per_block =
      threads_per_work_item * batch_size;

   const double faces_per_s =
      time_per_apply > 0.0
         ? static_cast< double >( num_faces_total ) / time_per_apply
         : 0.0;
   const double dofs_read_per_s =
      time_per_apply > 0.0
         ? static_cast< double >( dofs_read_per_apply ) / time_per_apply
         : 0.0;
   const double dofs_written_per_s =
      time_per_apply > 0.0
         ? static_cast< double >( dofs_written_per_apply ) /
              time_per_apply
         : 0.0;
   const double qpts_per_s =
      time_per_apply > 0.0
         ? static_cast< double >( qpts_per_apply ) / time_per_apply
         : 0.0;

   std::cout
      << "batched-global-face-advection,"
      << dim << ','
      << order << ','
      << num_quad_1d << ','
      << target_num_dofs << ','
      << actual_num_dofs << ','
      << num_cells << ','
      << num_faces_total << ','
      << face_counts[ 0 ] << ','
      << face_counts[ 1 ] << ','
      << face_counts[ 2 ] << ','
      << face_counts[ 3 ] << ','
      << face_counts[ 4 ] << ','
      << face_counts[ 5 ] << ','
      << actual_num_dofs << ','
      << extents[ 0 ] << ','
      << extents[ 1 ] << ','
      << extents[ 2 ] << ','
      << extents[ 3 ] << ','
      << extents[ 4 ] << ','
      << extents[ 5 ] << ','
      << layout << ','
      << threaded_dimensions << ','
      << kernel_family << ','
      << kernel << ','
      << face_policy << ','
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
      << faces_per_s << ','
      << dofs_read_per_s << ','
      << dofs_written_per_s << ','
      << qpts_per_s << ','
      << status << '\n';
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   size_t BatchSize >
void PrintSkippedDeviceGlobalFaceAdvectionRow(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions,
   const char * kernel_family,
   const char * kernel_name,
   const char * face_policy,
   const size_t target_threads_per_block,
   const size_t shared_memory_per_work_item,
   const char * status )
{
   static constexpr Integer num_quad_1d = Order + 2;

   const auto extents =
      GlobalFaceAdvectionBenchmarkExtents< Dim, Order >(
         target_num_dofs );
   const auto extent_columns = ExtentsAsSixColumns( extents );
   const GlobalIndex num_cells = Product( extents );
   const auto face_counts = CountFacesByDirection( extents );
   const GlobalIndex num_faces_total = SumSixColumns( face_counts );
   const GlobalIndex element_num_dofs =
      DofsPerElement< Dim, Order >();
   const GlobalIndex actual_num_dofs =
      num_cells * element_num_dofs;
   const GlobalIndex dofs_read_per_apply =
      2 * num_faces_total * element_num_dofs;
   const GlobalIndex dofs_written_per_apply =
      2 * num_faces_total * element_num_dofs;
   const GlobalIndex qpts_per_apply =
      num_faces_total *
      FaceQuadraturePointsPerFace< Dim, num_quad_1d >();
   const std::array< GlobalIndex, 3 > block_dims{
      ThreadLayout::GetNumberOfThreads(),
      BatchSize,
      1
   };

   PrintGlobalFaceAdvectionRow(
      Dim,
      Order,
      num_quad_1d,
      target_num_dofs,
      actual_num_dofs,
      num_cells,
      face_counts,
      extent_columns,
      layout_name,
      threaded_dimensions,
      kernel_family,
      kernel_name,
      face_policy,
      target_threads_per_block,
      ThreadLayout::GetNumberOfThreads(),
      BatchSize,
      block_dims,
      shared_memory_per_work_item,
      shared_memory_per_work_item * BatchSize,
      0.0,
      dofs_read_per_apply,
      dofs_written_per_apply,
      qpts_per_apply,
      status );
}

template < typename VectorType >
bool VectorFinite( const VectorType & values )
{
   for ( Integer i = 0; i < values.Size(); ++i )
   {
      if ( !std::isfinite( values[ i ] ) )
      {
         return false;
      }
   }
   return true;
}

template < typename VectorType >
bool VectorsClose(
   const VectorType & values,
   const VectorType & reference,
   const Real tolerance )
{
   Real diff_norm2 = 0.0;
   Real ref_norm2 = 0.0;
   Real max_abs_ref = 0.0;

   for ( Integer i = 0; i < values.Size(); ++i )
   {
      const Real diff = values[ i ] - reference[ i ];
      diff_norm2 += diff * diff;
      ref_norm2 += reference[ i ] * reference[ i ];
      const Real abs_ref = std::abs( reference[ i ] );
      max_abs_ref = abs_ref > max_abs_ref ? abs_ref : max_abs_ref;
   }

   const Real diff_norm = std::sqrt( diff_norm2 );
   const Real ref_norm = std::sqrt( ref_norm2 );
   const Real scale =
      ref_norm > Real{ 1.0 } ? ref_norm : Real{ 1.0 };

   return diff_norm <= tolerance * scale ||
          diff_norm <= tolerance * ( Real{ 1.0 } + max_abs_ref );
}

template < Integer Dim >
struct BenchmarkAdvectionVelocity
{
   GENDIL_HOST_DEVICE
   void operator()(
      const std::array< Real, Dim > & X,
      Real ( & velocity )[ Dim ] ) const
   {
      for ( Integer d = 0; d < Dim; ++d )
      {
         velocity[ d ] =
            0.5 +
            0.125 * static_cast< Real >( d + 1 ) +
            0.25 * X[ d ];
      }
   }
};

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename FaceMeshes,
   typename IntegrationRule,
   Integer Dim >
Vector ApplyGlobalFaceAdvectionOnce(
   const FiniteElementSpace & fe_space,
   const FaceMeshes & face_meshes,
   const IntegrationRule & integration_rule,
   const Vector & x,
   const Vector & baseline,
   BenchmarkAdvectionVelocity< Dim > adv )
{
   Vector y = baseline;
   auto op =
      MakeAdvectionFaceOperator< KernelPolicy >(
         fe_space,
         face_meshes,
         integration_rule,
         adv );
   op( x, y );
   GENDIL_DEVICE_SYNC;
   return y;
}

template <
   Integer Dim,
   Integer Order,
   typename KernelPolicy,
   typename ReferenceKernelPolicy,
   typename FiniteElementSpace,
   typename FaceMeshes,
   typename IntegrationRule >
bool CheckGlobalFaceAdvectionCorrectness(
   const FiniteElementSpace & fe_space,
   const FaceMeshes & face_meshes,
   const IntegrationRule & integration_rule,
   const GlobalIndex num_dofs )
{
   Vector x(
      static_cast< Integer >( num_dofs ),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.3125 +
            0.041 * static_cast< Real >( i ) +
            0.015625 * static_cast< Real >( ( i * 17 ) % 23 );
      } );
   Vector baseline(
      static_cast< Integer >( num_dofs ),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return -0.375 +
            0.0078125 * static_cast< Real >( ( i * 5 ) % 29 );
      } );

   auto reference =
      ApplyGlobalFaceAdvectionOnce<
         ReferenceKernelPolicy >(
            fe_space,
            face_meshes,
            integration_rule,
            x,
            baseline,
            BenchmarkAdvectionVelocity< Dim >{} );
   auto values =
      ApplyGlobalFaceAdvectionOnce<
         KernelPolicy >(
            fe_space,
            face_meshes,
            integration_rule,
            x,
            baseline,
            BenchmarkAdvectionVelocity< Dim >{} );

   return VectorFinite( reference ) &&
          VectorFinite( values ) &&
          VectorsClose(
             values,
             reference,
             global_face_advection_correctness_tolerance );
}

template <
   Integer Dim,
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename FaceMeshes,
   typename IntegrationRule >
double TimeGlobalFaceAdvection(
   const FiniteElementSpace & fe_space,
   const FaceMeshes & face_meshes,
   const IntegrationRule & integration_rule,
   const GlobalIndex num_dofs )
{
   Vector x(
      static_cast< Integer >( num_dofs ),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.25 +
            0.00001 * static_cast< Real >( i % 8191 );
      } );
   Vector y( static_cast< Integer >( num_dofs ) );
   y = 0.0;

   auto adv = BenchmarkAdvectionVelocity< Dim >{};
   auto op =
      MakeAdvectionFaceOperator< KernelPolicy >(
         fe_space,
         face_meshes,
         integration_rule,
         adv );

   return TimeOperator( op, x, y );
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
struct GlobalFaceAdvectionCaseSupported
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
   typename ReferenceKernelPolicy >
void RunGlobalFaceAdvectionKernelPolicy(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions,
   const char * kernel_family,
   const char * kernel_name,
   const char * face_policy,
   const size_t target_threads_per_block,
   const size_t batch_size )
{
   static constexpr Integer num_quad_1d = Order + 2;

   const auto extents =
      GlobalFaceAdvectionBenchmarkExtents< Dim, Order >(
         target_num_dofs );
   const auto extent_columns = ExtentsAsSixColumns( extents );
   const GlobalIndex num_cells = Product( extents );
   const auto face_counts = CountFacesByDirection( extents );
   const GlobalIndex num_faces_total = SumSixColumns( face_counts );
   const GlobalIndex element_num_dofs =
      DofsPerElement< Dim, Order >();
   const GlobalIndex actual_num_dofs =
      num_cells * element_num_dofs;
   const GlobalIndex dofs_read_per_apply =
      2 * num_faces_total * element_num_dofs;
   const GlobalIndex dofs_written_per_apply =
      2 * num_faces_total * element_num_dofs;
   const GlobalIndex qpts_per_apply =
      num_faces_total *
      FaceQuadraturePointsPerFace< Dim, num_quad_1d >();
   const auto block_dims =
      BlockDimensions< KernelPolicy >( num_faces_total );

   auto orders = MakeOrders< Dim, Order >();
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto mesh =
      MakeCartesianBenchmarkMeshFromGlobalExtents< Dim >( extents );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto face_meshes =
      make_cartesian_interior_face_connectivity< Dim >( extents );
   auto num_quads = MakeNumQuads< Dim, num_quad_1d >();
   auto integration_rule = MakeIntegrationRule( num_quads );

   const GlobalIndex counted_faces = CountGlobalFaces( face_meshes );
   if ( counted_faces != num_faces_total )
   {
      PrintGlobalFaceAdvectionRow(
         Dim,
         Order,
         num_quad_1d,
         target_num_dofs,
         actual_num_dofs,
         num_cells,
         face_counts,
         extent_columns,
         layout_name,
         threaded_dimensions,
         kernel_family,
         kernel_name,
         face_policy,
         target_threads_per_block,
         KernelPolicy::thread_layout_type::GetNumberOfThreads(),
         batch_size,
         block_dims,
         0,
         0,
         0.0,
         dofs_read_per_apply,
         dofs_written_per_apply,
         qpts_per_apply,
         "skipped-correctness" );
      return;
   }

   constexpr size_t required_shared_mem =
      global_face_advection_required_shared_memory_v<
         KernelPolicy,
         decltype( integration_rule ),
         decltype( fe_space ) >;
   constexpr size_t shared_memory_per_block =
      KernelContext<
         KernelPolicy,
         required_shared_mem >::shared_memory_block_size;

   if constexpr (
      !GlobalFaceAdvectionCaseSupported<
         KernelPolicy,
         decltype( fe_space ),
         decltype( integration_rule ) >::value )
   {
      PrintGlobalFaceAdvectionRow(
         Dim,
         Order,
         num_quad_1d,
         target_num_dofs,
         actual_num_dofs,
         num_cells,
         face_counts,
         extent_columns,
         layout_name,
         threaded_dimensions,
         kernel_family,
         kernel_name,
         face_policy,
         target_threads_per_block,
         KernelPolicy::thread_layout_type::GetNumberOfThreads(),
         batch_size,
         block_dims,
         required_shared_mem,
         shared_memory_per_block,
         0.0,
         dofs_read_per_apply,
         dofs_written_per_apply,
         qpts_per_apply,
         "skipped-helper-coverage" );
      return;
   }
   else
   {
      if ( !DeviceThreadLimitsFit( block_dims ) )
      {
         PrintGlobalFaceAdvectionRow(
            Dim,
            Order,
            num_quad_1d,
            target_num_dofs,
            actual_num_dofs,
            num_cells,
            face_counts,
            extent_columns,
            layout_name,
            threaded_dimensions,
            kernel_family,
            kernel_name,
            face_policy,
            target_threads_per_block,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            block_dims,
            required_shared_mem,
            shared_memory_per_block,
            0.0,
            dofs_read_per_apply,
            dofs_written_per_apply,
            qpts_per_apply,
            "skipped-launch-limit" );
         return;
      }

      if constexpr (
         shared_memory_per_block * sizeof( Real ) >
         static_shared_memory_compile_limit_bytes )
      {
         PrintGlobalFaceAdvectionRow(
            Dim,
            Order,
            num_quad_1d,
            target_num_dofs,
            actual_num_dofs,
            num_cells,
            face_counts,
            extent_columns,
            layout_name,
            threaded_dimensions,
            kernel_family,
            kernel_name,
            face_policy,
            target_threads_per_block,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            block_dims,
            required_shared_mem,
            shared_memory_per_block,
            0.0,
            dofs_read_per_apply,
            dofs_written_per_apply,
            qpts_per_apply,
            "skipped-shared-memory" );
         return;
      }
      else if ( !DeviceSharedMemoryFits( shared_memory_per_block ) )
      {
         PrintGlobalFaceAdvectionRow(
            Dim,
            Order,
            num_quad_1d,
            target_num_dofs,
            actual_num_dofs,
            num_cells,
            face_counts,
            extent_columns,
            layout_name,
            threaded_dimensions,
            kernel_family,
            kernel_name,
            face_policy,
            target_threads_per_block,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            block_dims,
            required_shared_mem,
            shared_memory_per_block,
            0.0,
            dofs_read_per_apply,
            dofs_written_per_apply,
            qpts_per_apply,
            "skipped-shared-memory" );
         return;
      }

      const bool correctness_ok =
         CheckGlobalFaceAdvectionCorrectness<
            Dim,
            Order,
            KernelPolicy,
            ReferenceKernelPolicy >(
               fe_space,
               face_meshes,
               integration_rule,
               actual_num_dofs );

      if ( !correctness_ok )
      {
         PrintGlobalFaceAdvectionRow(
            Dim,
            Order,
            num_quad_1d,
            target_num_dofs,
            actual_num_dofs,
            num_cells,
            face_counts,
            extent_columns,
            layout_name,
            threaded_dimensions,
            kernel_family,
            kernel_name,
            face_policy,
            target_threads_per_block,
            KernelPolicy::thread_layout_type::GetNumberOfThreads(),
            batch_size,
            block_dims,
            required_shared_mem,
            shared_memory_per_block,
            0.0,
            dofs_read_per_apply,
            dofs_written_per_apply,
            qpts_per_apply,
            "skipped-correctness" );
         return;
      }

      const double time_per_apply =
         TimeGlobalFaceAdvection<
            Dim,
            KernelPolicy >(
               fe_space,
               face_meshes,
               integration_rule,
               actual_num_dofs );
      PrintGlobalFaceAdvectionRow(
         Dim,
         Order,
         num_quad_1d,
         target_num_dofs,
         actual_num_dofs,
         num_cells,
         face_counts,
         extent_columns,
         layout_name,
         threaded_dimensions,
         kernel_family,
         kernel_name,
         face_policy,
         target_threads_per_block,
         KernelPolicy::thread_layout_type::GetNumberOfThreads(),
         batch_size,
         block_dims,
         required_shared_mem,
         shared_memory_per_block,
         time_per_apply,
         dofs_read_per_apply,
         dofs_written_per_apply,
         qpts_per_apply,
         "ok" );
   }
}

template < Integer Dim, Integer Order >
void RunSerialGlobalFaceAdvectionCase(
   const GlobalIndex target_num_dofs )
{
   RunGlobalFaceAdvectionKernelPolicy<
      Dim,
      Order,
      SerialKernelConfiguration,
      SerialKernelConfiguration >(
         target_num_dofs,
         "ThreadBlockLayout<>",
         0,
         "SerialKernelConfiguration",
         "SerialKernelConfiguration",
         "DirectGlobal",
         1,
         1 );
}

#if defined( GENDIL_USE_DEVICE )
template < typename ThreadLayout, size_t TargetThreads >
constexpr size_t GlobalFaceAdvectionBatchSizeForTarget()
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
   template < typename > typename FacePolicyTransform >
void RunLegacyGlobalFaceAdvectionCase(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions,
   const char * face_policy,
   const size_t skipped_shared_memory_per_work_item )
{
   if constexpr (
      ThreadLayout::GetNumberOfThreads() >
      max_threads_per_work_item )
   {
      PrintSkippedDeviceGlobalFaceAdvectionRow<
         Dim,
         Order,
         ThreadLayout,
         1 >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "ThreadFirstKernelConfiguration",
            "ThreadFirstKernelConfiguration",
            face_policy,
            ThreadLayout::GetNumberOfThreads(),
            skipped_shared_memory_per_work_item,
            "skipped-launch-limit" );
   }
   else
   {
      using KernelPolicy =
         FacePolicyTransform<
            ThreadFirstKernelConfiguration< ThreadLayout, Dim > >;

      RunGlobalFaceAdvectionKernelPolicy<
         Dim,
         Order,
         KernelPolicy,
         KernelPolicy >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "ThreadFirstKernelConfiguration",
            "ThreadFirstKernelConfiguration",
            face_policy,
            ThreadLayout::GetNumberOfThreads(),
            1 );
   }
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   template < typename > typename FacePolicyTransform >
void RunDeviceBatch1GlobalFaceAdvectionCase(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions,
   const char * face_policy,
   const size_t skipped_shared_memory_per_work_item )
{
   if constexpr (
      ThreadLayout::GetNumberOfThreads() >
      max_threads_per_work_item )
   {
      PrintSkippedDeviceGlobalFaceAdvectionRow<
         Dim,
         Order,
         ThreadLayout,
         1 >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "DeviceKernelConfigurationBatch1",
            "DeviceKernelConfiguration",
            face_policy,
            ThreadLayout::GetNumberOfThreads(),
            skipped_shared_memory_per_work_item,
            "skipped-launch-limit" );
   }
   else
   {
      using KernelPolicy =
         FacePolicyTransform<
            DeviceKernelConfiguration< ThreadLayout, Dim, 1 > >;

      RunGlobalFaceAdvectionKernelPolicy<
         Dim,
         Order,
         KernelPolicy,
         KernelPolicy >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "DeviceKernelConfigurationBatch1",
            "DeviceKernelConfiguration",
            face_policy,
            ThreadLayout::GetNumberOfThreads(),
            1 );
   }
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   template < typename > typename FacePolicyTransform,
   size_t BatchSize,
   size_t TargetThreads,
   bool FullSharedFacePolicy >
void RunDeviceBatchNGlobalFaceAdvectionCase(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions,
   const char * face_policy )
{
   constexpr size_t shared_memory_per_work_item =
      GlobalFaceAdvectionRequiredSharedMemoryEstimate<
         Dim,
         Order,
         FullSharedFacePolicy >();
   constexpr size_t shared_memory_per_block =
      shared_memory_per_work_item * BatchSize;
   constexpr size_t total_threads_per_block =
      ThreadLayout::GetNumberOfThreads() * BatchSize;

   if constexpr (
      ThreadLayout::GetNumberOfThreads() >
      max_threads_per_work_item )
   {
      PrintSkippedDeviceGlobalFaceAdvectionRow<
         Dim,
         Order,
         ThreadLayout,
         BatchSize >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "DeviceKernelConfigurationBatchN",
            "DeviceKernelConfiguration",
            face_policy,
            TargetThreads,
            shared_memory_per_work_item,
            "skipped-launch-limit" );
   }
   else if constexpr (
      total_threads_per_block >
      static_threads_per_block_compile_limit )
   {
      PrintSkippedDeviceGlobalFaceAdvectionRow<
         Dim,
         Order,
         ThreadLayout,
         BatchSize >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "DeviceKernelConfigurationBatchN",
            "DeviceKernelConfiguration",
            face_policy,
            TargetThreads,
            shared_memory_per_work_item,
            "skipped-launch-limit" );
   }
   else if constexpr (
      shared_memory_per_block * sizeof( Real ) >
      static_shared_memory_compile_limit_bytes )
   {
      PrintSkippedDeviceGlobalFaceAdvectionRow<
         Dim,
         Order,
         ThreadLayout,
         BatchSize >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "DeviceKernelConfigurationBatchN",
            "DeviceKernelConfiguration",
            face_policy,
            TargetThreads,
            shared_memory_per_work_item,
            "skipped-shared-memory" );
   }
   else
   {
      using KernelPolicy =
         FacePolicyTransform<
            DeviceKernelConfiguration< ThreadLayout, Dim, BatchSize > >;
      using ReferenceKernelPolicy =
         FacePolicyTransform<
            DeviceKernelConfiguration< ThreadLayout, Dim, 1 > >;

      RunGlobalFaceAdvectionKernelPolicy<
         Dim,
         Order,
         KernelPolicy,
         ReferenceKernelPolicy >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "DeviceKernelConfigurationBatchN",
            "DeviceKernelConfiguration",
            face_policy,
            TargetThreads,
            BatchSize );
   }
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   template < typename > typename FacePolicyTransform,
   size_t TargetThreads >
void RunDirectGlobalDeviceTarget(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions )
{
   constexpr size_t batch_size =
      GlobalFaceAdvectionBatchSizeForTarget<
         ThreadLayout,
         TargetThreads >();
   if constexpr ( batch_size > 1 )
   {
      RunDeviceBatchNGlobalFaceAdvectionCase<
         Dim,
         Order,
         ThreadLayout,
         FacePolicyTransform,
         batch_size,
         TargetThreads,
         false >(
            target_num_dofs,
            layout_name,
            threaded_dimensions,
            "DirectGlobal" );
   }
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout,
   template < typename > typename FacePolicyTransform >
void RunDirectGlobalDeviceTargetSweep(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions )
{
#if defined( GENDIL_USE_CUDA )
   RunDirectGlobalDeviceTarget<
      Dim,
      Order,
      ThreadLayout,
      FacePolicyTransform,
      32 >( target_num_dofs, layout_name, threaded_dimensions );
   if constexpr (
      GlobalFaceAdvectionBatchSizeForTarget< ThreadLayout, 64 >() !=
      GlobalFaceAdvectionBatchSizeForTarget< ThreadLayout, 32 >() )
   {
      RunDirectGlobalDeviceTarget<
         Dim,
         Order,
         ThreadLayout,
         FacePolicyTransform,
         64 >( target_num_dofs, layout_name, threaded_dimensions );
   }
#else
   RunDirectGlobalDeviceTarget<
      Dim,
      Order,
      ThreadLayout,
      FacePolicyTransform,
      64 >( target_num_dofs, layout_name, threaded_dimensions );
#endif

   if constexpr (
      GlobalFaceAdvectionBatchSizeForTarget< ThreadLayout, 128 >() !=
      GlobalFaceAdvectionBatchSizeForTarget< ThreadLayout, 64 >() )
   {
      RunDirectGlobalDeviceTarget<
         Dim,
         Order,
         ThreadLayout,
         FacePolicyTransform,
         128 >( target_num_dofs, layout_name, threaded_dimensions );
   }
   if constexpr (
      GlobalFaceAdvectionBatchSizeForTarget< ThreadLayout, 256 >() !=
      GlobalFaceAdvectionBatchSizeForTarget< ThreadLayout, 128 >() )
   {
      RunDirectGlobalDeviceTarget<
         Dim,
         Order,
         ThreadLayout,
         FacePolicyTransform,
         256 >( target_num_dofs, layout_name, threaded_dimensions );
   }
   if constexpr (
      GlobalFaceAdvectionBatchSizeForTarget< ThreadLayout, 512 >() !=
      GlobalFaceAdvectionBatchSizeForTarget< ThreadLayout, 256 >() )
   {
      RunDirectGlobalDeviceTarget<
         Dim,
         Order,
         ThreadLayout,
         FacePolicyTransform,
         512 >( target_num_dofs, layout_name, threaded_dimensions );
   }
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunDirectGlobalLayout(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions )
{
   RunLegacyGlobalFaceAdvectionCase<
      Dim,
      Order,
      ThreadLayout,
      DefaultFaceDofPolicies >(
         target_num_dofs,
         layout_name,
         threaded_dimensions,
         "DirectGlobal",
         GlobalFaceAdvectionRequiredSharedMemoryEstimate<
            Dim,
            Order,
            false >() );
   RunDeviceBatch1GlobalFaceAdvectionCase<
      Dim,
      Order,
      ThreadLayout,
      DefaultFaceDofPolicies >(
         target_num_dofs,
         layout_name,
         threaded_dimensions,
         "DirectGlobal",
         GlobalFaceAdvectionRequiredSharedMemoryEstimate<
            Dim,
            Order,
            false >() );
   RunDirectGlobalDeviceTargetSweep<
      Dim,
      Order,
      ThreadLayout,
      DefaultFaceDofPolicies >(
         target_num_dofs,
         layout_name,
         threaded_dimensions );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunDirectGlobalRegisterOnlyLayout(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions )
{
   RunDeviceBatch1GlobalFaceAdvectionCase<
      Dim,
      Order,
      ThreadLayout,
      DefaultFaceDofPolicies >(
         target_num_dofs,
         layout_name,
         threaded_dimensions,
         "DirectGlobal",
         GlobalFaceAdvectionRequiredSharedMemoryEstimate<
            Dim,
            Order,
            false >() );
   RunDirectGlobalDeviceTargetSweep<
      Dim,
      Order,
      ThreadLayout,
      DefaultFaceDofPolicies >(
         target_num_dofs,
         layout_name,
         threaded_dimensions );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunFullSharedRepresentativeLayout(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions )
{
   RunLegacyGlobalFaceAdvectionCase<
      Dim,
      Order,
      ThreadLayout,
      FullSharedFaceDofPolicies >(
         target_num_dofs,
         layout_name,
         threaded_dimensions,
         "FullShared",
         GlobalFaceAdvectionRequiredSharedMemoryEstimate<
            Dim,
            Order,
            true >() );
   RunDeviceBatch1GlobalFaceAdvectionCase<
      Dim,
      Order,
      ThreadLayout,
      FullSharedFaceDofPolicies >(
         target_num_dofs,
         layout_name,
         threaded_dimensions,
         "FullShared",
         GlobalFaceAdvectionRequiredSharedMemoryEstimate<
            Dim,
            Order,
            true >() );
   RunDeviceBatchNGlobalFaceAdvectionCase<
      Dim,
      Order,
      ThreadLayout,
      FullSharedFaceDofPolicies,
      device_warp_size,
      ThreadLayout::GetNumberOfThreads() * device_warp_size,
      true >(
         target_num_dofs,
         layout_name,
         threaded_dimensions,
         "FullShared" );
}

template <
   Integer Dim,
   Integer Order,
   typename ThreadLayout >
void RunFullSharedRegisterOnlyRepresentativeLayout(
   const GlobalIndex target_num_dofs,
   const char * layout_name,
   const size_t threaded_dimensions )
{
   RunDeviceBatch1GlobalFaceAdvectionCase<
      Dim,
      Order,
      ThreadLayout,
      FullSharedFaceDofPolicies >(
         target_num_dofs,
         layout_name,
         threaded_dimensions,
         "FullShared",
         GlobalFaceAdvectionRequiredSharedMemoryEstimate<
            Dim,
            Order,
            true >() );
   RunDeviceBatchNGlobalFaceAdvectionCase<
      Dim,
      Order,
      ThreadLayout,
      FullSharedFaceDofPolicies,
      device_warp_size,
      ThreadLayout::GetNumberOfThreads() * device_warp_size,
      true >(
         target_num_dofs,
         layout_name,
         threaded_dimensions,
         "FullShared" );
}
#endif

template < Integer Dim, Integer Order >
void RunGlobalFaceAdvectionTensorProductLayouts(
   const GlobalIndex target_num_dofs )
{
#if defined( GENDIL_USE_DEVICE )
   RunDirectGlobalRegisterOnlyLayout< Dim, Order, ThreadBlockLayout<> >(
      target_num_dofs,
      "ThreadBlockLayout<>",
      0 );
   RunDirectGlobalLayout< Dim, Order, ThreadBlockLayout< Order + 2 > >(
      target_num_dofs,
      "ThreadBlockLayout<num_quad_1d>",
      1 );

   if constexpr ( Dim >= 2 )
   {
      RunDirectGlobalLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d>",
            2 );
   }

   if constexpr ( Dim >= 3 )
   {
      RunDirectGlobalLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2, Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>",
            3 );
   }

   if constexpr ( Dim >= 4 )
   {
      RunDirectGlobalLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            4 );
   }

   if constexpr ( Dim >= 5 )
   {
      RunDirectGlobalLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            5 );
   }

   if constexpr ( Dim >= 6 )
   {
      RunDirectGlobalLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            6 );
   }

   RunFullSharedRegisterOnlyRepresentativeLayout<
      Dim,
      Order,
      ThreadBlockLayout<> >(
      target_num_dofs,
      "ThreadBlockLayout<>",
      0 );
   RunFullSharedRepresentativeLayout<
      Dim,
      Order,
      ThreadBlockLayout< Order + 2 > >(
         target_num_dofs,
         "ThreadBlockLayout<num_quad_1d>",
         1 );

   if constexpr ( Dim >= 2 )
   {
      RunFullSharedRepresentativeLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d>",
            2 );
   }

   if constexpr ( Dim >= 3 )
   {
      RunFullSharedRepresentativeLayout<
         Dim,
         Order,
         ThreadBlockLayout< Order + 2, Order + 2, Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>",
            3 );
   }

   if constexpr ( Dim >= 4 )
   {
      RunFullSharedRepresentativeLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            4 );
   }

   if constexpr ( Dim >= 5 )
   {
      RunFullSharedRepresentativeLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            5 );
   }

   if constexpr ( Dim >= 6 )
   {
      RunFullSharedRepresentativeLayout<
         Dim,
         Order,
         ThreadBlockLayout<
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2,
            Order + 2 > >(
            target_num_dofs,
            "ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>",
            6 );
   }
#endif
}

template < Integer Dim, Integer Order >
void RunGlobalFaceAdvectionOrder(
   const GlobalIndex target_num_dofs )
{
   RunSerialGlobalFaceAdvectionCase< Dim, Order >( target_num_dofs );
   RunGlobalFaceAdvectionTensorProductLayouts< Dim, Order >(
      target_num_dofs );
}

template < Integer Dim >
void RunGlobalFaceAdvectionDimension(
   const GlobalIndex target_num_dofs )
{
   RunGlobalFaceAdvectionOrder< Dim, 0 >( target_num_dofs );
   RunGlobalFaceAdvectionOrder< Dim, 1 >( target_num_dofs );
   RunGlobalFaceAdvectionOrder< Dim, 2 >( target_num_dofs );
   RunGlobalFaceAdvectionOrder< Dim, 3 >( target_num_dofs );
}

} // namespace gendil::benchmarks
