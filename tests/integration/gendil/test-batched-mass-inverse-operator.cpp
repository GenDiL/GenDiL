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
      << "test-batched-mass-inverse-operator skipped because "
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
Real L2Norm( const VectorType & x )
{
   Real norm_sq = 0.0;
   for ( Integer i = 0; i < x.Size(); ++i )
   {
      norm_sq += x[ i ] * x[ i ];
   }
   return std::sqrt( norm_sq );
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

template < typename A, typename B >
bool CheckClose(
   const char * label,
   const A & observed,
   const B & expected,
   const Real tolerance )
{
   const Real abs_error = AbsoluteL2Error( observed, expected );
   const Real scale = std::max( Real{ 1.0 }, L2Norm( expected ) );
   const Real scaled_error = abs_error / scale;

   std::cout << label << " scaled L2 error = " << scaled_error << '\n';
   if (
      !std::isfinite( abs_error ) ||
      !std::isfinite( scaled_error ) ||
      scaled_error > tolerance )
   {
      std::cout << "FAILED: " << label << " exceeded tolerance "
                << tolerance << ".\n";
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
         return 0.5 +
            0.03125 * static_cast< Real >( i ) +
            0.13 * static_cast< Real >( ( i * 7 ) % 19 );
      } );
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule,
   typename Sigma >
Vector ApplyMassInverse(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   Sigma & sigma,
   const Vector & rhs,
   const Integer max_iters = 10000,
   const Real tolerance = 1e-14 )
{
   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   x = 0.0;

   auto op =
      MakeMassInverseFiniteElementOperator< KernelPolicy >(
         fe_space,
         integration_rule,
         sigma,
         max_iters,
         tolerance );
   op( rhs, x );
   GENDIL_DEVICE_SYNC;

   return x;
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule,
   typename Sigma >
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

template < Integer BatchSize >
bool ShouldRunResidualCheck( const GlobalIndex num_cells )
{
   return ( BatchSize == 2 || BatchSize == 4 ) &&
          ( num_cells == 1 || num_cells == BatchSize + 1 );
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   bool RunResidualCheck = true >
bool RunMassInverseCaseForCellCount(
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
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 1.25 + 0.25 * x + 0.5 * x * x;
   };

   Vector rhs = MakeInputVector( fe_space.GetNumberOfFiniteElementDofs() );

   auto x_legacy =
      ApplyMassInverse< LegacyConfig >(
         fe_space,
         integration_rule,
         sigma,
         rhs );
   auto x_batch1 =
      ApplyMassInverse< DeviceBatch1 >(
         fe_space,
         integration_rule,
         sigma,
         rhs );
   auto x_batchn =
      ApplyMassInverse< DeviceBatchN >(
         fe_space,
         integration_rule,
         sigma,
         rhs );

   constexpr Real tolerance = 1e-8;
   bool success = true;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   success =
      CheckClose(
         "DeviceBatchN vs LegacyConfig",
         x_batchn,
         x_legacy,
         tolerance ) && success;
   success =
      CheckClose(
         "DeviceBatchN vs DeviceBatch1",
         x_batchn,
         x_batch1,
         tolerance ) && success;
   success =
      CheckClose(
         "DeviceBatch1 vs LegacyConfig",
         x_batch1,
         x_legacy,
         tolerance ) && success;

   if constexpr ( BatchSize == 2 )
   {
      if ( num_cells == 1 )
      {
         auto x_explicit_defaults =
            ApplyMassInverse< DeviceBatchN >(
               fe_space,
               integration_rule,
               sigma,
               rhs,
               10000,
               1e-14 );
         success =
            CheckClose(
               "DeviceBatchN explicit defaults vs implicit defaults",
               x_explicit_defaults,
               x_batchn,
               tolerance ) && success;
      }
   }

   if constexpr ( RunResidualCheck )
   {
      if ( ShouldRunResidualCheck< BatchSize >( num_cells ) )
      {
         auto residual =
            ApplyMass< DeviceBatchN >(
               fe_space,
               integration_rule,
               sigma,
               x_batchn );
         success =
            CheckClose(
               "Mass(DeviceBatchN inverse result) vs rhs",
               residual,
               rhs,
               tolerance ) && success;
      }
   }

   return success;
}

template < Integer BatchSize >
bool RunRegisterOnlyMassInverseCaseForCellCount(
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
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 1.0 + x + 0.25 * x * x;
   };

   Vector rhs = MakeInputVector( fe_space.GetNumberOfFiniteElementDofs() );

   auto x_batch1 =
      ApplyMassInverse< DeviceBatch1 >(
         fe_space,
         integration_rule,
         sigma,
         rhs );
   auto x_batchn =
      ApplyMassInverse< DeviceBatchN >(
         fe_space,
         integration_rule,
         sigma,
         rhs );

   constexpr Real tolerance = 1e-8;
   bool success = true;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   success =
      CheckClose(
         "Register-only DeviceBatchN vs DeviceBatch1",
         x_batchn,
         x_batch1,
         tolerance ) && success;

   if constexpr ( BatchSize == 2 )
   {
      if ( num_cells == 1 )
      {
         auto x_explicit_defaults =
            ApplyMassInverse< DeviceBatchN >(
               fe_space,
               integration_rule,
               sigma,
               rhs,
               10000,
               1e-14 );
         success =
            CheckClose(
               "Register-only explicit defaults vs implicit defaults",
               x_explicit_defaults,
               x_batchn,
               tolerance ) && success;
      }
   }

   if ( ShouldRunResidualCheck< BatchSize >( num_cells ) )
   {
      auto residual =
         ApplyMass< DeviceBatchN >(
            fe_space,
            integration_rule,
            sigma,
            x_batchn );
      success =
         CheckClose(
            "Register-only Mass(DeviceBatchN inverse result) vs rhs",
            residual,
            rhs,
            tolerance ) && success;
   }

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
   bool RunResidualCheck = true >
bool RunThreadedMassInverseBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunMassInverseCaseForCellCount<
            Layout,
            MaxSharedDimensions,
            BatchSize,
            RunResidualCheck >( label, num_cells );
      } );
}

template < Integer BatchSize >
bool RunRegisterOnlyMassInverseBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunRegisterOnlyMassInverseCaseForCellCount< BatchSize >(
            label,
            num_cells );
      } );
}

bool TestThreadedMassInverse()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 4 );

   bool success = true;
   success =
      RunThreadedMassInverseBatchCases< Layout, MaxSharedDimensions, 1 >(
         "ThreadBlockLayout<4>, BatchSize=1" ) && success;
   success =
      RunThreadedMassInverseBatchCases< Layout, MaxSharedDimensions, 2 >(
         "ThreadBlockLayout<4>, BatchSize=2" ) && success;
   success =
      RunThreadedMassInverseBatchCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<4>, BatchSize=4" ) && success;
   success =
      RunThreadedMassInverseBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size >(
            "ThreadBlockLayout<4>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestRegisterOnlyMassInverse()
{
   bool success = true;
   success =
      RunRegisterOnlyMassInverseBatchCases< 1 >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunRegisterOnlyMassInverseBatchCases< 2 >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunRegisterOnlyMassInverseBatchCases< 4 >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunRegisterOnlyMassInverseBatchCases< device_warp_size >(
         "ThreadBlockLayout<>, BatchSize=device_warp_size" ) && success;
   return success;
}

bool TestIrregularMassInverseDiagnostic()
{
   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static_assert( Layout::GetNumberOfThreads() == 15 );

   std::cout
      << "Running diagnostic irregular ThreadBlockLayout<3,5> "
      << "MassInverse cases. For this 1D operator, extra logical thread "
      << "dimensions are expected to be idle.\n";

   bool success = true;
   success =
      RunThreadedMassInverseBatchCases<
         Layout,
         MaxSharedDimensions,
         1,
         false >(
         "ThreadBlockLayout<3,5>, BatchSize=1 diagnostic" ) && success;
   success =
      RunThreadedMassInverseBatchCases<
         Layout,
         MaxSharedDimensions,
         2,
         false >(
         "ThreadBlockLayout<3,5>, BatchSize=2 diagnostic" ) && success;
   success =
      RunThreadedMassInverseBatchCases<
         Layout,
         MaxSharedDimensions,
         4,
         false >(
         "ThreadBlockLayout<3,5>, BatchSize=4 diagnostic" ) && success;
   success =
      RunThreadedMassInverseBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         false >(
            "ThreadBlockLayout<3,5>, BatchSize=device_warp_size diagnostic" ) &&
      success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestThreadedMassInverse() && success;
   success = TestRegisterOnlyMassInverse() && success;
   success = TestIrregularMassInverseDiagnostic() && success;

   return success ? 0 : 1;
}

#endif
