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
      << "test-batched-mixed-mass-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{

template <
   typename KernelPolicy,
   typename TrialFiniteElementSpace,
   typename TestFiniteElementSpace,
   typename Rule,
   typename Sigma >
Vector ApplyMixedMass(
   const TrialFiniteElementSpace & trial_fe_space,
   const TestFiniteElementSpace & test_fe_space,
   const Rule & integration_rule,
   Sigma & sigma,
   const Vector & x )
{
   Vector y( test_fe_space.GetNumberOfFiniteElementDofs() );
   y = 0.0;

   auto op =
      MakeMixedMassOperator< KernelPolicy >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         sigma );
   op( x, y );
   GENDIL_DEVICE_SYNC;

   return y;
}

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool RunThreadedMixedMassCaseForCellCount(
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

   static constexpr Integer trial_order = 2;
   static constexpr Integer test_order = 3;
   static constexpr Integer num_quad_1d = 5;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< trial_order > trial_orders;
   auto trial_finite_element = MakeLobattoFiniteElement( trial_orders );
   auto trial_fe_space =
      MakeFiniteElementSpace( mesh, trial_finite_element );

   FiniteElementOrders< test_order > test_orders;
   auto test_finite_element = MakeLobattoFiniteElement( test_orders );
   auto test_fe_space =
      MakeFiniteElementSpace( mesh, test_finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 0.875 + 0.5 * x + x * x;
   };

   Vector x(
      trial_fe_space.GetNumberOfFiniteElementDofs(),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.0625 +
            0.041 * static_cast< Real >( i ) +
            0.19 * static_cast< Real >( ( i * 3 ) % 7 );
      } );

   auto y_legacy =
      ApplyMixedMass< LegacyConfig >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         sigma,
         x );
   auto y_batch1 =
      ApplyMixedMass< DeviceBatch1 >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         sigma,
         x );
   auto y_batchn =
      ApplyMixedMass< DeviceBatchN >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         sigma,
         x );

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
bool RunRegisterOnlyMixedMassCaseForCellCount(
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

   static constexpr Integer trial_order = 2;
   static constexpr Integer test_order = 3;
   static constexpr Integer num_quad_1d = 5;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< trial_order > trial_orders;
   auto trial_finite_element = MakeLobattoFiniteElement( trial_orders );
   auto trial_fe_space =
      MakeFiniteElementSpace( mesh, trial_finite_element );

   FiniteElementOrders< test_order > test_orders;
   auto test_finite_element = MakeLobattoFiniteElement( test_orders );
   auto test_fe_space =
      MakeFiniteElementSpace( mesh, test_finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 1.0 + 0.25 * x + 0.75 * x * x;
   };

   Vector x(
      trial_fe_space.GetNumberOfFiniteElementDofs(),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.125 +
            0.03125 * static_cast< Real >( i ) +
            0.17 * static_cast< Real >( ( i * 7 ) % 11 );
      } );

   auto y_batch1 =
      ApplyMixedMass< DeviceBatch1 >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         sigma,
         x );
   auto y_batchn =
      ApplyMixedMass< DeviceBatchN >(
         trial_fe_space,
         test_fe_space,
         integration_rule,
         sigma,
         x );

   constexpr Real tolerance = 1e-10;
   std::cout << label << ", num_cells = " << num_cells << '\n';
   return CheckScaledL2Close(
      "Register-only DeviceBatchN vs DeviceBatch1",
      y_batchn,
      y_batch1,
      tolerance );
}

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool RunThreadedMixedMassBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunThreadedMixedMassCaseForCellCount<
            Layout,
            MaxSharedDimensions,
            BatchSize >( label, num_cells );
      } );
}

template < Integer BatchSize >
bool RunRegisterOnlyMixedMassBatchCases( const char * label )
{
   return RunNormalizedCellCases< BatchSize >(
      [=] ( const GlobalIndex num_cells )
      {
         return RunRegisterOnlyMixedMassCaseForCellCount< BatchSize >(
            label,
            num_cells );
      } );
}

bool TestThreadedMixedMass()
{
   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;

   bool success = true;
   success =
      RunThreadedMixedMassBatchCases< Layout, MaxSharedDimensions, 1 >(
         "ThreadBlockLayout<num_quad_1d>, BatchSize=1" ) && success;
   success =
      RunThreadedMixedMassBatchCases< Layout, MaxSharedDimensions, 2 >(
         "ThreadBlockLayout<num_quad_1d>, BatchSize=2" ) && success;
   success =
      RunThreadedMixedMassBatchCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<num_quad_1d>, BatchSize=4" ) && success;
   success =
      RunThreadedMixedMassBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size >(
            "ThreadBlockLayout<num_quad_1d>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestRegisterOnlyMixedMass()
{
   bool success = true;
   success =
      RunRegisterOnlyMixedMassBatchCases< 1 >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunRegisterOnlyMixedMassBatchCases< 2 >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunRegisterOnlyMixedMassBatchCases< 4 >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunRegisterOnlyMixedMassBatchCases< device_warp_size >(
         "ThreadBlockLayout<>, BatchSize=device_warp_size" ) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestThreadedMixedMass() && success;
   success = TestRegisterOnlyMixedMass() && success;

   return success ? 0 : 1;
}

#endif
