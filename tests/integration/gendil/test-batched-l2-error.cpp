// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-l2-error skipped because GENDIL_USE_DEVICE "
      << "is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{
constexpr Real sentinel = -12345.25;

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

Vector MakeInputVector( const Integer size )
{
   return Vector(
      size,
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.375 +
            0.04125 * static_cast< Real >( i ) +
            0.19 * static_cast< Real >( ( i * 7 ) % 17 );
      } );
}

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule >
Real ComputeL2Error(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & x )
{
   auto exact =
      [] GENDIL_HOST_DEVICE ( const std::array< Real, 1 > & X ) -> Real
      {
         const Real x = X[ 0 ];
         return 0.125 + 0.875 * x - 0.3125 * x * x + 0.0625 * x * x * x;
      };

   const Real error =
      L2Error< KernelPolicy >( fe_space, integration_rule, exact, x );
   GENDIL_DEVICE_SYNC;

   return error;
}

bool CheckClose(
   const char * label,
   const Real observed,
   const Real expected,
   const Real tolerance )
{
   const Real abs_error = std::abs( observed - expected );
   const Real scale = std::max( Real{ 1.0 }, std::abs( expected ) );
   const Real scaled_error = abs_error / scale;

   std::cout << label << " observed = " << observed
             << ", expected = " << expected
             << ", scaled error = " << scaled_error << '\n';

   if (
      !std::isfinite( observed ) ||
      !std::isfinite( expected ) ||
      !std::isfinite( scaled_error ) ||
      scaled_error > tolerance )
   {
      std::cout << "FAILED: " << label << " exceeded tolerance "
                << tolerance << ".\n";
      return false;
   }
   return true;
}

bool CheckNontrivial( const char * label, const Real value )
{
   if ( !std::isfinite( value ) || value <= 1e-8 )
   {
      std::cout << "FAILED: " << label
                << " produced a near-zero oracle, which would not detect "
                << "duplicate inactive-lane contributions reliably.\n";
      return false;
   }
   return true;
}

template < typename Config >
bool CheckZeroWorkItems( const char * label )
{
   DeviceBuffer< Real > marker( 1, sentinel );
   auto marker_data = marker.data;

   Config::BlockLoop(
      0,
      [=] GENDIL_HOST_DEVICE ( Config ) mutable
      {
         marker_data[ 0 ] = 1.0;
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

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize >
bool RunL2ErrorCaseForCellCount(
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
   static constexpr Integer num_quad_1d = order + 2;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   Vector x = MakeInputVector( fe_space.GetNumberOfFiniteElementDofs() );

   const Real error_legacy =
      ComputeL2Error< LegacyConfig >( fe_space, integration_rule, x );
   const Real error_batch1 =
      ComputeL2Error< DeviceBatch1 >( fe_space, integration_rule, x );
   const Real error_batchn =
      ComputeL2Error< DeviceBatchN >( fe_space, integration_rule, x );

   constexpr Real tolerance = 1e-10;
   bool success = true;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   success =
      CheckNontrivial( "LegacyConfig L2Error", error_legacy ) && success;
   success =
      CheckClose(
         "DeviceBatchN vs LegacyConfig",
         error_batchn,
         error_legacy,
         tolerance ) && success;
   success =
      CheckClose(
         "DeviceBatchN vs DeviceBatch1",
         error_batchn,
         error_batch1,
         tolerance ) && success;
   success =
      CheckClose(
         "DeviceBatch1 vs LegacyConfig",
         error_batch1,
         error_legacy,
         tolerance ) && success;
   return success;
}

template < Integer BatchSize >
bool RunRegisterOnlyL2ErrorCaseForCellCount(
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
   static constexpr Integer num_quad_1d = order + 2;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   Vector x = MakeInputVector( fe_space.GetNumberOfFiniteElementDofs() );

   const Real error_batch1 =
      ComputeL2Error< DeviceBatch1 >( fe_space, integration_rule, x );
   const Real error_batchn =
      ComputeL2Error< DeviceBatchN >( fe_space, integration_rule, x );

   constexpr Real tolerance = 1e-10;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   return
      CheckNontrivial( "DeviceBatch1 L2Error", error_batch1 ) &&
      CheckClose(
         "Register-only DeviceBatchN vs DeviceBatch1",
         error_batchn,
         error_batch1,
         tolerance );
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
   Integer BatchSize >
bool RunThreadedL2ErrorBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunL2ErrorCaseForCellCount<
            Layout,
            MaxSharedDimensions,
            BatchSize >( label, num_cells );
      } );
}

template < Integer BatchSize >
bool RunRegisterOnlyL2ErrorBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunRegisterOnlyL2ErrorCaseForCellCount< BatchSize >(
            label,
            num_cells );
      } );
}

bool TestThreadedL2Error()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 4 );

   bool success = true;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 1 >(
         "ThreadBlockLayout<4>, BatchSize=1" ) && success;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 2 >(
         "ThreadBlockLayout<4>, BatchSize=2" ) && success;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<4>, BatchSize=4" ) && success;
   success =
      RunThreadedL2ErrorBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size >(
            "ThreadBlockLayout<4>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestIrregularL2Error()
{
   std::cout
      << "Running diagnostic irregular ThreadBlockLayout<3,5> L2Error "
      << "cases. For this 1D operator, extra logical thread dimensions are "
      << "expected to be idle.\n";

   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static_assert( Layout::GetNumberOfThreads() == 15 );

   bool success = true;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 1 >(
         "ThreadBlockLayout<3,5>, BatchSize=1 diagnostic" ) && success;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 2 >(
         "ThreadBlockLayout<3,5>, BatchSize=2 diagnostic" ) && success;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<3,5>, BatchSize=4 diagnostic" ) && success;
   success =
      RunThreadedL2ErrorBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size >(
            "ThreadBlockLayout<3,5>, BatchSize=device_warp_size diagnostic" )
      && success;
   return success;
}

bool TestRegisterOnlyL2Error()
{
   bool success = true;
   success =
      RunRegisterOnlyL2ErrorBatchCases< 1 >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunRegisterOnlyL2ErrorBatchCases< 2 >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunRegisterOnlyL2ErrorBatchCases< 4 >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunRegisterOnlyL2ErrorBatchCases< device_warp_size >(
         "ThreadBlockLayout<>, BatchSize=device_warp_size" ) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRegisterOnlyL2Error() && success;
   success = TestThreadedL2Error() && success;
   success = TestIrregularL2Error() && success;

   return success ? 0 : 1;
}

#endif
