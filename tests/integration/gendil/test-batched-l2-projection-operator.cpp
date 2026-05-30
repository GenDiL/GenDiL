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
      << "test-batched-l2-projection-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{
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
      CheckRelativeL2Close(
         "DeviceBatchN vs LegacyConfig",
         y_batchn,
         y_legacy,
         tolerance ) && success;
   success =
      CheckRelativeL2Close(
         "DeviceBatchN vs DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;
   success =
      CheckRelativeL2Close(
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
   return CheckRelativeL2Close(
      "Register-only DeviceBatchN vs DeviceBatch1",
      y_batchn,
      y_batch1,
      tolerance );
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
