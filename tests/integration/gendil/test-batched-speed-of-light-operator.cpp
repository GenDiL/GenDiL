// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-speed-of-light-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule >
Vector ApplySpeedOfLight(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & x )
{
   Vector y( fe_space.GetNumberOfFiniteElementDofs() );
   y = 0.0;

   auto op =
      MakeSpeedOfLightOperator< KernelPolicy >(
         fe_space,
         integration_rule );
   op( x, y );
   GENDIL_DEVICE_SYNC;

   return y;
}

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool RunThreadedSpeedOfLightCaseForCellCount(
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
         CheckZeroWorkItems< LegacyConfig >(
            label,
            integer_sentinel ) && success;
      success =
         CheckZeroWorkItems< DeviceBatch1 >(
            label,
            integer_sentinel ) && success;
      success =
         CheckZeroWorkItems< DeviceBatchN >(
            label,
            integer_sentinel ) && success;
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

   Vector x(
      fe_space.GetNumberOfFiniteElementDofs(),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.25 +
            0.0625 * static_cast< Real >( i ) +
            0.125 * static_cast< Real >( ( i * 5 ) % 13 );
      } );

   auto y_legacy =
      ApplySpeedOfLight< LegacyConfig >( fe_space, integration_rule, x );
   auto y_batch1 =
      ApplySpeedOfLight< DeviceBatch1 >( fe_space, integration_rule, x );
   auto y_batchn =
      ApplySpeedOfLight< DeviceBatchN >( fe_space, integration_rule, x );

   constexpr Real tolerance = 1e-10;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   bool success = true;
   success =
      CheckScaledL2Close(
         "DeviceBatchN vs LegacyConfig",
         y_batchn,
         y_legacy,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "DeviceBatchN vs DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "DeviceBatch1 vs LegacyConfig",
         y_batch1,
         y_legacy,
         tolerance ) && success;
   return success;
}

template < Integer BatchSize >
bool RunRegisterOnlySpeedOfLightCaseForCellCount(
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
         CheckZeroWorkItems< DeviceBatch1 >(
            label,
            integer_sentinel ) && success;
      success =
         CheckZeroWorkItems< DeviceBatchN >(
            label,
            integer_sentinel ) && success;
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

   Vector x(
      fe_space.GetNumberOfFiniteElementDofs(),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.125 +
            0.03125 * static_cast< Real >( i ) +
            0.17 * static_cast< Real >( ( i * 7 ) % 11 );
      } );

   auto y_batch1 =
      ApplySpeedOfLight< DeviceBatch1 >( fe_space, integration_rule, x );
   auto y_batchn =
      ApplySpeedOfLight< DeviceBatchN >( fe_space, integration_rule, x );

   constexpr Real tolerance = 1e-10;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   return CheckScaledL2Close(
      "Register-only DeviceBatchN vs DeviceBatch1",
      y_batchn,
      y_batch1,
      tolerance );
}

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool RunThreadedSpeedOfLightBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunThreadedSpeedOfLightCaseForCellCount<
            Layout,
            MaxSharedDimensions,
            BatchSize >( label, num_cells );
      } );
}

template < Integer BatchSize >
bool RunRegisterOnlySpeedOfLightBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunRegisterOnlySpeedOfLightCaseForCellCount< BatchSize >(
            label,
            num_cells );
      } );
}

bool TestThreadedSpeedOfLight()
{
   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;

   bool success = true;
   success =
      RunThreadedSpeedOfLightBatchCases< Layout, MaxSharedDimensions, 1 >(
         "ThreadBlockLayout<num_quad_1d>, BatchSize=1" ) && success;
   success =
      RunThreadedSpeedOfLightBatchCases< Layout, MaxSharedDimensions, 2 >(
         "ThreadBlockLayout<num_quad_1d>, BatchSize=2" ) && success;
   success =
      RunThreadedSpeedOfLightBatchCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<num_quad_1d>, BatchSize=4" ) && success;
   success =
      RunThreadedSpeedOfLightBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size >(
            "ThreadBlockLayout<num_quad_1d>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestRegisterOnlySpeedOfLight()
{
   bool success = true;
   success =
      RunRegisterOnlySpeedOfLightBatchCases< 1 >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunRegisterOnlySpeedOfLightBatchCases< 2 >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunRegisterOnlySpeedOfLightBatchCases< 4 >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunRegisterOnlySpeedOfLightBatchCases< device_warp_size >(
         "ThreadBlockLayout<>, BatchSize=device_warp_size" ) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestThreadedSpeedOfLight() && success;
   success = TestRegisterOnlySpeedOfLight() && success;

   return success ? 0 : 1;
}

#endif
