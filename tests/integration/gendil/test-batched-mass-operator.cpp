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
      << "test-batched-mass-operator skipped because GENDIL_USE_DEVICE "
      << "is not enabled.\n";
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

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule, typename Sigma >
Vector ApplyMass(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   Sigma & sigma,
   const Vector & x )
{
   Vector y( fe_space.GetNumberOfFiniteElementDofs() );
   y = 0.0;

   auto op =
      MakeMassFiniteElementOperator< KernelPolicy >(
         fe_space,
         integration_rule,
         sigma );
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
      [=] GENDIL_DEVICE ( const Config & ) mutable
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

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool RunMassCaseForCellCount(
   const char * label,
   const GlobalIndex num_cells )
{
   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return true;
   }

   using LegacyConfig =
      ThreadFirstKernelConfiguration< Layout, MaxSharedDimensions >;
   using DeviceBatchN =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   if ( num_cells == 0 )
   {
      bool success = true;
      success = CheckZeroWorkItems< LegacyConfig >( label ) && success;
      success = CheckZeroWorkItems< DeviceBatchN >( label ) && success;
      return success;
   }

   static constexpr Integer order = 3;
   static constexpr Integer num_quad_1d = order + 1;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 1.0 + 0.25 * x + x * x;
   };

   Vector x(
      fe_space.GetNumberOfFiniteElementDofs(),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.125 +
            0.03125 * static_cast< Real >( i ) +
            0.17 * static_cast< Real >( ( i * 7 ) % 11 );
      } );

   auto y_legacy =
      ApplyMass< LegacyConfig >( fe_space, integration_rule, sigma, x );
   auto y_batchn =
      ApplyMass< DeviceBatchN >( fe_space, integration_rule, sigma, x );

   constexpr Real tolerance = 1e-12;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   return CheckClose(
      "DeviceBatchN vs LegacyConfig",
      y_batchn,
      y_legacy,
      tolerance );
}

template < Integer BatchSize >
bool RunRegisterOnlyMassCaseForCellCount(
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
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 0.75 + x + 0.5 * x * x;
   };

   Vector x(
      fe_space.GetNumberOfFiniteElementDofs(),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.125 +
            0.03125 * static_cast< Real >( i ) +
            0.17 * static_cast< Real >( ( i * 7 ) % 11 );
      } );

   auto y_batch1 =
      ApplyMass< DeviceBatch1 >( fe_space, integration_rule, sigma, x );
   auto y_batchn =
      ApplyMass< DeviceBatchN >( fe_space, integration_rule, sigma, x );

   constexpr Real tolerance = 1e-12;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   return CheckClose(
      "Register-only DeviceBatchN vs DeviceBatch1",
      y_batchn,
      y_batch1,
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

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool RunThreadedMassBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunMassCaseForCellCount<
            Layout,
            MaxSharedDimensions,
            BatchSize >( label, num_cells );
      } );
}

template < Integer BatchSize >
bool RunRegisterOnlyMassBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunRegisterOnlyMassCaseForCellCount< BatchSize >(
            label,
            num_cells );
      } );
}

bool TestThreadedMass()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 4 );

   bool success = true;
   success =
      RunThreadedMassBatchCases< Layout, MaxSharedDimensions, 1 >(
         "ThreadBlockLayout<4>, BatchSize=1" ) && success;
   success =
      RunThreadedMassBatchCases< Layout, MaxSharedDimensions, 2 >(
         "ThreadBlockLayout<4>, BatchSize=2" ) && success;
   success =
      RunThreadedMassBatchCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<4>, BatchSize=4" ) && success;
   success =
      RunThreadedMassBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size >(
            "ThreadBlockLayout<4>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestRegisterOnlyMass()
{
   bool success = true;
   success =
      RunRegisterOnlyMassBatchCases< 1 >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunRegisterOnlyMassBatchCases< 2 >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunRegisterOnlyMassBatchCases< 4 >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunRegisterOnlyMassBatchCases< device_warp_size >(
         "ThreadBlockLayout<>, BatchSize=device_warp_size" ) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestThreadedMass() && success;
   success = TestRegisterOnlyMass() && success;

   return success ? 0 : 1;
}

#endif
