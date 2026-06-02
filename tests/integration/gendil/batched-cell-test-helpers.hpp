// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef GENDIL_TESTS_INTEGRATION_GENDIL_BATCHED_CELL_TEST_HELPERS_HPP
#define GENDIL_TESTS_INTEGRATION_GENDIL_BATCHED_CELL_TEST_HELPERS_HPP

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace gendil::test
{

inline constexpr long long integer_sentinel = -987654321;
inline constexpr Real real_sentinel = -12345.25;

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

inline Integer GetDeviceMaxThreadsPerBlock()
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

template < typename VectorType >
Real RelativeL2Error( const VectorType & a, const VectorType & b )
{
   const Real abs_err = AbsoluteL2Error( a, b );
   const Real norm_b = L2Norm( b );

   if ( norm_b == 0.0 )
   {
      return abs_err;
   }
   return abs_err / norm_b;
}

template < typename A, typename B >
bool CheckRelativeL2Close(
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

template < typename A, typename B >
bool CheckScaledL2Close(
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

inline bool CheckScalarClose(
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

inline bool CheckNontrivialPositive( const char * label, const Real value )
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

template < typename Config, typename T >
bool CheckZeroWorkItems(
   const char * label,
   const T sentinel_value,
   const T write_value = T{ 1 } )
{
   DeviceBuffer< T > marker( 1, sentinel_value );
   auto marker_data = marker.data;

   Config::BlockLoop(
      0,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex ) mutable
      {
         marker_data[ 0 ] = write_value;
      } );
   GENDIL_DEVICE_SYNC;

   marker.CopyToHost();

   const bool success = marker.data.host_pointer[ 0 ] == sentinel_value;
   if ( !success )
   {
      std::cout << "FAILED: zero-work-item launch wrote in "
                << label << ".\n";
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

template < Integer BatchSize, typename Lambda >
bool RunConservativeCellCases( Lambda && run_case )
{
   bool success = true;
   success = run_case( GlobalIndex{ 1 } ) && success;
   success = run_case( GlobalIndex{ BatchSize + 1 } ) && success;
   return success;
}

inline Vector MakeConstantVector( const Integer size, const Real value )
{
   Vector x( size );
   x = value;
   return x;
}

inline Vector MakeBaselineVector( const Integer size )
{
   return Vector(
      size,
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 10.0 +
            0.125 * static_cast< Real >( i ) +
            0.03125 * static_cast< Real >( ( i * 5 ) % 7 );
      } );
}

inline Vector AddVectors( const Vector & a, const Vector & b )
{
   GENDIL_VERIFY( a.Size() == b.Size(), "Vector sizes do not match." );

   Vector result( a.Size() );
   Real * result_data = result.WriteHostData();
   const Real * a_data = a.ReadHostData();
   const Real * b_data = b.ReadHostData();
   for ( Integer i = 0; i < result.Size(); ++i )
   {
      result_data[ i ] = a_data[ i ] + b_data[ i ];
   }
   return result;
}

inline bool CheckNoValue(
   const char * label,
   const Vector & x,
   const Real forbidden_value,
   const Real tolerance = 1.0e-12 )
{
   bool success = true;
   const Real * data = x.ReadHostData();
   for ( Integer i = 0; i < x.Size(); ++i )
   {
      if ( std::abs( data[ i ] - forbidden_value ) < tolerance )
      {
         std::cout << "FAILED: " << label << " left sentinel at index "
                   << i << ".\n";
         success = false;
      }
   }
   return success;
}

} // namespace gendil::test

#endif // GENDIL_TESTS_INTEGRATION_GENDIL_BATCHED_CELL_TEST_HELPERS_HPP
