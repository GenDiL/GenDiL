// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <array>
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
using namespace gendil::test;

namespace
{
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
      success =
         CheckZeroWorkItems< LegacyConfig >( label, real_sentinel ) &&
         success;
      success =
         CheckZeroWorkItems< DeviceBatch1 >( label, real_sentinel ) &&
         success;
      success =
         CheckZeroWorkItems< DeviceBatchN >( label, real_sentinel ) &&
         success;
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
      CheckNontrivialPositive( "LegacyConfig L2Error", error_legacy ) &&
      success;
   success =
      CheckScalarClose(
         "DeviceBatchN vs LegacyConfig",
         error_batchn,
         error_legacy,
         tolerance ) && success;
   success =
      CheckScalarClose(
         "DeviceBatchN vs DeviceBatch1",
         error_batchn,
         error_batch1,
         tolerance ) && success;
   success =
      CheckScalarClose(
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
      success =
         CheckZeroWorkItems< DeviceBatch1 >( label, real_sentinel ) &&
         success;
      success =
         CheckZeroWorkItems< DeviceBatchN >( label, real_sentinel ) &&
         success;
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
      CheckNontrivialPositive( "DeviceBatch1 L2Error", error_batch1 ) &&
      CheckScalarClose(
         "Register-only DeviceBatchN vs DeviceBatch1",
         error_batchn,
         error_batch1,
         tolerance );
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
   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 5 );

   bool success = true;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 1 >(
         "ThreadBlockLayout<5>, BatchSize=1" ) && success;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 2 >(
         "ThreadBlockLayout<5>, BatchSize=2" ) && success;
   success =
      RunThreadedL2ErrorBatchCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<5>, BatchSize=4" ) && success;
   success =
      RunThreadedL2ErrorBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size >(
            "ThreadBlockLayout<5>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestIrregularL2Error()
{
   std::cout
      << "Skipping ThreadBlockLayout<3,5> L2Error diagnostic: the current "
      << "threaded helper contract requires the mapped 1D thread dimension "
      << "to cover the local DOF/quadrature extent.\n";
   return true;
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
