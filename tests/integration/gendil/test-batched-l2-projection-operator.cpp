// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-l2-projection-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{
constexpr long long sentinel = -987654321;

template < typename T >
struct DeviceBuffer
{
   HostDevicePointer< T > data;
   GlobalIndex size;

   DeviceBuffer( const GlobalIndex n, const T value )
      : size( n )
   {
      AllocateHostPointer( size, data );
      AllocateDevicePointer( size, data );
      Fill( value );
   }

   DeviceBuffer( const DeviceBuffer & ) = delete;
   DeviceBuffer & operator=( const DeviceBuffer & ) = delete;

   ~DeviceBuffer()
   {
      FreeDevicePointer( data );
      FreeHostPointer( data );
   }

   void Fill( const T value )
   {
      for ( GlobalIndex i = 0; i < size; ++i )
      {
         data.host_pointer[ i ] = value;
      }
      ToDevice( size, data );
   }

   void CopyToHost() const
   {
      ToHost( size, data );
   }
};

Integer GetDeviceMaxThreadsPerBlock()
{
#if defined( GENDIL_USE_CUDA )
   int device = 0;
   cudaDeviceProp properties;
   GENDIL_GPU_CHECK( cudaGetDevice( &device ) );
   GENDIL_GPU_CHECK( cudaGetDeviceProperties( &properties, device ) );
   return static_cast< Integer >( properties.maxThreadsPerBlock );
#elif defined( GENDIL_USE_HIP )
   int device = 0;
   hipDeviceProp_t properties;
   GENDIL_GPU_CHECK( hipGetDevice( &device ) );
   GENDIL_GPU_CHECK( hipGetDeviceProperties( &properties, device ) );
   return static_cast< Integer >( properties.maxThreadsPerBlock );
#else
   return 1;
#endif
}

template < typename Layout, Integer BatchSize >
bool LaunchConfigurationFits( const char * label )
{
   const Integer requested_threads =
      Layout::GetNumberOfThreads() * BatchSize;
   const Integer max_threads = GetDeviceMaxThreadsPerBlock();

   if ( requested_threads > max_threads )
   {
      std::cout << "Skipping " << label
                << ": requested " << requested_threads
                << " threads per block, device limit is "
                << max_threads << ".\n";
      return false;
   }
   return true;
}

template < typename VectorType >
Real AbsoluteL2Error( const VectorType & a, const VectorType & b )
{
   GENDIL_VERIFY( a.Size() == b.Size(), "Vector sizes do not match." );

   Real err_sq = 0.0;
   for ( Integer i = 0; i < a.Size(); ++i )
   {
      const Real d = a[ i ] - b[ i ];
      err_sq += d * d;
   }
   return std::sqrt( err_sq );
}

template < typename VectorType >
Real RelativeL2Error( const VectorType & a, const VectorType & b )
{
   const Real abs_err = AbsoluteL2Error( a, b );

   Real norm_b_sq = 0.0;
   for ( Integer i = 0; i < b.Size(); ++i )
   {
      norm_b_sq += b[ i ] * b[ i ];
   }

   const Real norm_b = std::sqrt( norm_b_sq );
   if ( norm_b == 0.0 )
   {
      return abs_err;
   }
   return abs_err / norm_b;
}

template <
   typename TrialKernelPolicy,
   typename TestKernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename Rule >
Vector ApplyL2Projection(
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const Rule & integration_rule,
   const Vector & x )
{
   Vector y( test_fe_space.GetNumberOfFiniteElementDofs() );
   y = 0.0;

   auto op =
      MakeL2Projection< TrialKernelPolicy, TestKernelPolicy >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         integration_rule );
   op( x, y );
   GENDIL_DEVICE_SYNC;

   return y;
}

template < typename A, typename B >
bool CheckClose(
   const char * label,
   const A & observed,
   const B & expected,
   const Real tolerance )
{
   const Real rel_error = RelativeL2Error( observed, expected );
   std::cout << label << " relative L2 error = " << rel_error << '\n';
   if ( rel_error > tolerance )
   {
      std::cout << "FAILED: " << label << " exceeded tolerance "
                << tolerance << ".\n";
      return false;
   }
   return true;
}

template < typename Config >
bool CheckZeroWorkItems( const char * label )
{
   DeviceBuffer< long long > marker( 1, sentinel );
   auto marker_data = marker.data;

   Config::BlockLoop(
      0,
      [=] GENDIL_HOST_DEVICE ( Config ) mutable
      {
         marker_data[ 0 ] = 1;
      } );
   GENDIL_DEVICE_SYNC;

   marker.CopyToHost();

   const bool success = marker.data.host_pointer[ 0 ] == sentinel;
   if ( !success )
   {
      std::cout << "FAILED: zero-work-item launch wrote in "
                << label << ".\n";
   }
   return success;
}

Vector MakeInputVector( const Integer size )
{
   return Vector(
      size,
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.25 +
            0.0625 * static_cast< Real >( i ) +
            0.11 * static_cast< Real >( ( i * 5 ) % 13 );
      } );
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize >
bool RunL2ProjectionCaseForCellCount(
   const char * label,
   const GlobalIndex num_cells )
{
   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return true;
   }

   using LegacyConfig =
      ThreadFirstKernelConfiguration< Layout, MaxSharedDimensions >;
   using DeviceBatch1 =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchN =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   if ( num_cells == 0 )
   {
      bool success = true;
      success = CheckZeroWorkItems< LegacyConfig >( label ) && success;
      success = CheckZeroWorkItems< DeviceBatch1 >( label ) && success;
      success = CheckZeroWorkItems< DeviceBatchN >( label ) && success;
      return success;
   }

   static constexpr Integer order = 3;
   static constexpr Integer num_quad_1d = order + 1;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh line_mesh( h, num_cells );
   auto mesh = MakeCartesianProductMesh( line_mesh );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto trial_fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto test_fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   Vector x = MakeInputVector(
      trial_fe_space.GetNumberOfFiniteElementDofs() );

   auto y_legacy =
      ApplyL2Projection< LegacyConfig, LegacyConfig >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         x );
   auto y_batch1 =
      ApplyL2Projection< DeviceBatch1, DeviceBatch1 >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         x );
   auto y_batchn =
      ApplyL2Projection< DeviceBatchN, DeviceBatchN >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         x );

   constexpr Real tolerance = 1e-10;
   bool success = true;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   success =
      CheckClose(
         "DeviceBatchN vs LegacyConfig",
         y_batchn,
         y_legacy,
         tolerance ) && success;
   success =
      CheckClose(
         "DeviceBatchN vs DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;
   success =
      CheckClose(
         "DeviceBatch1 vs LegacyConfig",
         y_batch1,
         y_legacy,
         tolerance ) && success;
   return success;
}

template < Integer BatchSize >
bool RunRegisterOnlyL2ProjectionCaseForCellCount(
   const char * label,
   const GlobalIndex num_cells )
{
   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 0;

   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return true;
   }

   using DeviceBatch1 =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchN =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   if ( num_cells == 0 )
   {
      bool success = true;
      success = CheckZeroWorkItems< DeviceBatch1 >( label ) && success;
      success = CheckZeroWorkItems< DeviceBatchN >( label ) && success;
      return success;
   }

   static constexpr Integer order = 3;
   static constexpr Integer num_quad_1d = order + 1;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh line_mesh( h, num_cells );
   auto mesh = MakeCartesianProductMesh( line_mesh );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto trial_fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto test_fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   Vector x = MakeInputVector(
      trial_fe_space.GetNumberOfFiniteElementDofs() );

   auto y_batch1 =
      ApplyL2Projection< DeviceBatch1, DeviceBatch1 >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         x );
   auto y_batchn =
      ApplyL2Projection< DeviceBatchN, DeviceBatchN >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         x );

   constexpr Real tolerance = 1e-10;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   return CheckClose(
      "Register-only DeviceBatchN vs DeviceBatch1",
      y_batchn,
      y_batch1,
      tolerance );
}

template < Integer BatchSize, typename Lambda >
bool RunConservativeCellCases( Lambda && run_case )
{
   bool success = true;
   success = run_case( GlobalIndex{ 1 } ) && success;
   success = run_case( GlobalIndex{ BatchSize + 1 } ) && success;
   return success;
}

template < Integer BatchSize, typename Lambda >
bool RunNormalizedCellCases( Lambda && run_case )
{
   const std::array< GlobalIndex, 6 > candidates{
      0,
      1,
      BatchSize - 1,
      BatchSize,
      BatchSize + 1,
      10
   };
   std::array< GlobalIndex, 6 > cases{};
   Integer num_cases = 0;

   for ( const GlobalIndex candidate : candidates )
   {
      bool found = false;
      for ( Integer i = 0; i < num_cases; ++i )
      {
         found = found || cases[ i ] == candidate;
      }
      if ( !found )
      {
         cases[ num_cases++ ] = candidate;
      }
   }

   bool success = true;
   for ( Integer i = 0; i < num_cases; ++i )
   {
      success = run_case( cases[ i ] ) && success;
   }
   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   bool FullMatrix >
bool RunThreadedL2ProjectionBatchCases(
   const char * label )
{
   auto run_case =
      [=] ( const GlobalIndex num_cells )
      {
         return RunL2ProjectionCaseForCellCount<
            Layout,
            MaxSharedDimensions,
            BatchSize >( label, num_cells );
      };

   if constexpr ( FullMatrix )
   {
      return RunNormalizedCellCases< BatchSize >( run_case );
   }
   else
   {
      return RunConservativeCellCases< BatchSize >( run_case );
   }
}

template < Integer BatchSize, bool FullMatrix >
bool RunRegisterOnlyL2ProjectionBatchCases(
   const char * label )
{
   auto run_case =
      [=] ( const GlobalIndex num_cells )
      {
         return RunRegisterOnlyL2ProjectionCaseForCellCount< BatchSize >(
            label,
            num_cells );
      };

   if constexpr ( FullMatrix )
   {
      return RunNormalizedCellCases< BatchSize >( run_case );
   }
   else
   {
      return RunConservativeCellCases< BatchSize >( run_case );
   }
}

bool TestConservativeL2Projection()
{
   std::cout << "Running conservative batched L2 projection matrix.\n";

   using ThreadedLayout = ThreadBlockLayout< 4 >;
   static constexpr Integer ThreadedMaxSharedDimensions = 1;

   bool success = true;
   success =
      RunRegisterOnlyL2ProjectionBatchCases< 2, false >(
         "ThreadBlockLayout<>, BatchSize=2 conservative" ) && success;
   success =
      RunRegisterOnlyL2ProjectionBatchCases< 4, false >(
         "ThreadBlockLayout<>, BatchSize=4 conservative" ) && success;
   success =
      RunThreadedL2ProjectionBatchCases<
         ThreadedLayout,
         ThreadedMaxSharedDimensions,
         2,
         false >(
            "ThreadBlockLayout<4>, BatchSize=2 conservative" ) && success;
   success =
      RunThreadedL2ProjectionBatchCases<
         ThreadedLayout,
         ThreadedMaxSharedDimensions,
         4,
         false >(
            "ThreadBlockLayout<4>, BatchSize=4 conservative" ) && success;
   return success;
}

bool TestExpandedThreadedL2Projection()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 4 );

   bool success = true;
   success =
      RunThreadedL2ProjectionBatchCases<
         Layout,
         MaxSharedDimensions,
         1,
         true >(
            "ThreadBlockLayout<4>, BatchSize=1" ) && success;
   success =
      RunThreadedL2ProjectionBatchCases<
         Layout,
         MaxSharedDimensions,
         2,
         true >(
            "ThreadBlockLayout<4>, BatchSize=2" ) && success;
   success =
      RunThreadedL2ProjectionBatchCases<
         Layout,
         MaxSharedDimensions,
         4,
         true >(
            "ThreadBlockLayout<4>, BatchSize=4" ) && success;
   success =
      RunThreadedL2ProjectionBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         true >(
            "ThreadBlockLayout<4>, BatchSize=device_warp_size" ) && success;
   return success;
}

bool TestExpandedIrregularL2Projection()
{
   std::cout
      << "Running diagnostic irregular ThreadBlockLayout<3,5> cases. "
      << "For this 1D operator, extra logical thread dimensions are expected "
      << "to be idle.\n";

   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static_assert( Layout::GetNumberOfThreads() == 15 );

   bool success = true;
   success =
      RunThreadedL2ProjectionBatchCases<
         Layout,
         MaxSharedDimensions,
         1,
         true >(
            "ThreadBlockLayout<3,5>, BatchSize=1 diagnostic" ) && success;
   success =
      RunThreadedL2ProjectionBatchCases<
         Layout,
         MaxSharedDimensions,
         2,
         true >(
            "ThreadBlockLayout<3,5>, BatchSize=2 diagnostic" ) && success;
   success =
      RunThreadedL2ProjectionBatchCases<
         Layout,
         MaxSharedDimensions,
         4,
         true >(
            "ThreadBlockLayout<3,5>, BatchSize=4 diagnostic" ) && success;
   success =
      RunThreadedL2ProjectionBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         true >(
            "ThreadBlockLayout<3,5>, BatchSize=device_warp_size diagnostic" )
      && success;
   return success;
}

bool TestExpandedRegisterOnlyL2Projection()
{
   bool success = true;
   success =
      RunRegisterOnlyL2ProjectionBatchCases< 1, true >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunRegisterOnlyL2ProjectionBatchCases< 2, true >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunRegisterOnlyL2ProjectionBatchCases< 4, true >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunRegisterOnlyL2ProjectionBatchCases< device_warp_size, true >(
         "ThreadBlockLayout<>, BatchSize=device_warp_size" ) && success;
   return success;
}

} // namespace

int main()
{
   if ( !TestConservativeL2Projection() )
   {
      return 1;
   }

   bool success = true;
   success = TestExpandedRegisterOnlyL2Projection() && success;
   success = TestExpandedThreadedL2Projection() && success;
   success = TestExpandedIrregularL2Projection() && success;

   return success ? 0 : 1;
}

#endif
