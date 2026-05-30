// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <utility>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-linear-form-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{
constexpr long long sentinel = -987654321;
constexpr Real output_sentinel = -123456.75;

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

Vector MakeConstantVector( const Integer size, const Real value )
{
   Vector x( size );
   x = value;
   return x;
}

Vector MakeBaselineVector( const Integer size )
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

Vector AddVectors( const Vector & a, const Vector & b )
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

bool CheckNoSentinel( const char * label, const Vector & x )
{
   bool success = true;
   const Real * data = x.ReadHostData();
   for ( Integer i = 0; i < x.Size(); ++i )
   {
      if ( std::abs( data[ i ] - output_sentinel ) < 1.0e-12 )
      {
         std::cout << "FAILED: " << label << " left sentinel at index "
                   << i << ".\n";
         success = false;
      }
   }
   return success;
}

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule, typename Source >
Vector MakeLinearFormVector(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   Source & source )
{
   Vector rhs =
      MakeLinearForm< KernelPolicy >(
         fe_space,
         integration_rule,
         source );
   GENDIL_DEVICE_SYNC;
   return rhs;
}

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule, typename Source >
Vector ApplyLinearFormWithInitial(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   Source & source,
   const Vector & initial )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using MeshQuadData =
      typename Mesh::cell_type::template QuadData< Rule >;
   using ElementQuadData =
      decltype(
         MakeDofToQuad<
            typename FiniteElementSpace::finite_element_type::shape_functions,
            Rule >() );

   Vector out( initial );
   MeshQuadData mesh_quad_data;
   ElementQuadData element_quad_data;
   auto dofs_out =
      MakeReadWriteElementTensorView< KernelPolicy >( fe_space, out );

   LinearFormOperator< KernelPolicy, Rule >(
      fe_space,
      mesh_quad_data,
      element_quad_data,
      source,
      dofs_out );
   GENDIL_DEVICE_SYNC;

   return out;
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

void FillH1LineRestriction(
   const GlobalIndex num_cells,
   HostDevicePointer< int > & indices )
{
   const GlobalIndex entries = 2 * num_cells;
   AllocateHostPointer( entries, indices );
   AllocateDevicePointer( entries, indices );

   for ( GlobalIndex e = 0; e < num_cells; ++e )
   {
      indices.host_pointer[ 2 * e ] = static_cast< int >( e );
      indices.host_pointer[ 2 * e + 1 ] = static_cast< int >( e + 1 );
   }
   ToDevice( entries, indices );
}

HostDevicePointer< const int > MakeConstRestrictionView(
   const HostDevicePointer< int > & indices )
{
   HostDevicePointer< const int > view;
   view.host_pointer = indices.host_pointer;
#if defined( GENDIL_USE_DEVICE )
   view.device_pointer = indices.device_pointer;
#endif
   return view;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   bool CompareLegacy >
bool RunL2CaseForCellCount(
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
      if constexpr ( CompareLegacy )
      {
         success = CheckZeroWorkItems< LegacyConfig >( label ) && success;
      }
      success = CheckZeroWorkItems< DeviceBatch1 >( label ) && success;
      success = CheckZeroWorkItems< DeviceBatchN >( label ) && success;
      return success;
   }

   static constexpr Integer order = 3;
   static constexpr Integer num_quad_1d = order + 2;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto source = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 0.75 + 0.5 * x + 0.125 * x * x;
   };

   auto rhs_batch1 =
      MakeLinearFormVector< DeviceBatch1 >(
         fe_space,
         integration_rule,
         source );
   auto rhs_batchn =
      MakeLinearFormVector< DeviceBatchN >(
         fe_space,
         integration_rule,
         source );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   std::cout << label << " L2, num_cells = " << num_cells << '\n';

   if constexpr ( CompareLegacy )
   {
      auto rhs_legacy =
         MakeLinearFormVector< LegacyConfig >(
            fe_space,
            integration_rule,
            source );
      success =
         CheckClose(
            "DeviceBatchN vs LegacyConfig",
            rhs_batchn,
            rhs_legacy,
            tolerance ) && success;
      success =
         CheckClose(
            "DeviceBatch1 vs LegacyConfig",
            rhs_batch1,
            rhs_legacy,
            tolerance ) && success;
   }

   success =
      CheckClose(
         "DeviceBatchN vs DeviceBatch1",
         rhs_batchn,
         rhs_batch1,
         tolerance ) && success;

   auto zero = MakeConstantVector( fe_space.GetNumberOfFiniteElementDofs(), 0.0 );
   auto sentinel_initial =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         output_sentinel );
   auto direct_zero =
      ApplyLinearFormWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         source,
         zero );
   auto direct_sentinel =
      ApplyLinearFormWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         source,
         sentinel_initial );

   success =
      CheckClose(
         "DeviceBatchN direct zero-baseline vs MakeLinearForm",
         direct_zero,
         rhs_batchn,
         tolerance ) && success;
   success =
      CheckClose(
         "DeviceBatchN L2 overwrite from sentinel",
         direct_sentinel,
         direct_zero,
         tolerance ) && success;
   success = CheckNoSentinel( "DeviceBatchN L2 overwrite", direct_sentinel ) &&
             success;

   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   bool CompareLegacy >
bool RunH1CaseForCellCount(
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
      if constexpr ( CompareLegacy )
      {
         success = CheckZeroWorkItems< LegacyConfig >( label ) && success;
      }
      success = CheckZeroWorkItems< DeviceBatch1 >( label ) && success;
      success = CheckZeroWorkItems< DeviceBatchN >( label ) && success;
      return success;
   }

   static constexpr Integer order = 1;
   static constexpr Integer num_quad_1d = order + 3;

   HostDevicePointer< int > indices;
   FillH1LineRestriction( num_cells, indices );
   auto const_indices = MakeConstRestrictionView( indices );
   H1Restriction restriction{
      const_indices,
      static_cast< Integer >( num_cells + 1 )
   };

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element, restriction );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto source = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 1.125 + 0.375 * x + 0.625 * x * x;
   };

   auto rhs_batch1 =
      MakeLinearFormVector< DeviceBatch1 >(
         fe_space,
         integration_rule,
         source );
   auto rhs_batchn =
      MakeLinearFormVector< DeviceBatchN >(
         fe_space,
         integration_rule,
         source );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   std::cout << label << " H1, num_cells = " << num_cells << '\n';

   if constexpr ( CompareLegacy )
   {
      auto rhs_legacy =
         MakeLinearFormVector< LegacyConfig >(
            fe_space,
            integration_rule,
            source );
      success =
         CheckClose(
            "DeviceBatchN vs LegacyConfig",
            rhs_batchn,
            rhs_legacy,
            tolerance ) && success;
      success =
         CheckClose(
            "DeviceBatch1 vs LegacyConfig",
            rhs_batch1,
            rhs_legacy,
            tolerance ) && success;
   }

   success =
      CheckClose(
         "DeviceBatchN vs DeviceBatch1",
         rhs_batchn,
         rhs_batch1,
         tolerance ) && success;

   auto zero = MakeConstantVector( fe_space.GetNumberOfFiniteElementDofs(), 0.0 );
   auto baseline = MakeBaselineVector( fe_space.GetNumberOfFiniteElementDofs() );
   auto direct_zero =
      ApplyLinearFormWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         source,
         zero );
   auto direct_baseline =
      ApplyLinearFormWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         source,
         baseline );
   auto expected_accumulated = AddVectors( baseline, direct_zero );

   success =
      CheckClose(
         "DeviceBatchN direct zero-baseline vs MakeLinearForm",
         direct_zero,
         rhs_batchn,
         tolerance ) && success;
   success =
      CheckClose(
         "DeviceBatchN H1 accumulation from baseline",
         direct_baseline,
         expected_accumulated,
         tolerance ) && success;

   FreeDevicePointer( indices );
   FreeHostPointer( indices );

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
   bool CompareLegacy >
bool RunLinearFormBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         bool success = true;
         success =
            RunL2CaseForCellCount<
               Layout,
               MaxSharedDimensions,
               BatchSize,
               CompareLegacy >( label, num_cells ) && success;
         success =
            RunH1CaseForCellCount<
               Layout,
               MaxSharedDimensions,
               BatchSize,
               CompareLegacy >( label, num_cells ) && success;
         return success;
      } );
}

bool TestThreadedLinearForm()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 4 );

   bool success = true;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 1, true >(
         "ThreadBlockLayout<4>, BatchSize=1" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 2, true >(
         "ThreadBlockLayout<4>, BatchSize=2" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 4, true >(
         "ThreadBlockLayout<4>, BatchSize=4" ) && success;
   success =
      RunLinearFormBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         true >(
            "ThreadBlockLayout<4>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestRegisterOnlyLinearForm()
{
   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 0;

   bool success = true;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 1, false >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 2, false >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 4, false >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunLinearFormBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         false >(
            "ThreadBlockLayout<>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestIrregularLinearFormDiagnostic()
{
   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static_assert( Layout::GetNumberOfThreads() == 15 );

   std::cout
      << "Running diagnostic irregular ThreadBlockLayout<3,5> "
      << "LinearForm cases. For this 1D operator, extra logical thread "
      << "dimensions are expected to be idle.\n";

   bool success = true;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 1, true >(
         "ThreadBlockLayout<3,5>, BatchSize=1 diagnostic" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 2, true >(
         "ThreadBlockLayout<3,5>, BatchSize=2 diagnostic" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 4, true >(
         "ThreadBlockLayout<3,5>, BatchSize=4 diagnostic" ) && success;
   success =
      RunLinearFormBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         true >(
            "ThreadBlockLayout<3,5>, BatchSize=device_warp_size diagnostic" ) &&
      success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestThreadedLinearForm() && success;
   success = TestRegisterOnlyLinearForm() && success;
   success = TestIrregularLinearFormDiagnostic() && success;

   return success ? 0 : 1;
}

#endif
