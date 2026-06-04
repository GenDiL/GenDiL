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
      << "test-batched-mass-operator skipped because GENDIL_USE_DEVICE "
      << "is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{
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
      success =
         CheckZeroWorkItems< LegacyConfig >(
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
   return CheckRelativeL2Close(
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
   return CheckRelativeL2Close(
      "Register-only DeviceBatchN vs DeviceBatch1",
      y_batchn,
      y_batch1,
      tolerance );
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
