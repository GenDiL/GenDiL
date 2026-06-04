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
      << "test-batched-mass-inverse-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
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
      CheckScaledL2Close(
         "DeviceBatchN vs LegacyConfig",
         x_batchn,
         x_legacy,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "DeviceBatchN vs DeviceBatch1",
         x_batchn,
         x_batch1,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
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
            CheckScaledL2Close(
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
            CheckScaledL2Close(
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
      CheckScaledL2Close(
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
            CheckScaledL2Close(
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
         CheckScaledL2Close(
            "Register-only Mass(DeviceBatchN inverse result) vs rhs",
            residual,
            rhs,
            tolerance ) && success;
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
   std::cout
      << "Skipping ThreadBlockLayout<3,5> MassInverse diagnostic: the "
      << "current threaded helper contract requires the mapped 1D thread "
      << "dimension to cover the local DOF/quadrature extent.\n";
   return true;
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
