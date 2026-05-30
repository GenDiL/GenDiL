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
      << "test-batched-grad-grad-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{
constexpr Real output_sentinel = -123456.75;

Vector MakeInputVector( const Integer size )
{
   return Vector(
      size,
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.3125 +
            0.07125 * static_cast< Real >( i ) +
            0.17 * static_cast< Real >( ( i * 7 ) % 13 ) +
            0.011 * static_cast< Real >( ( i * i ) % 19 );
      } );
}

Vector Make2DInputVector( const Integer size )
{
   return Vector(
      size,
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.1875 +
            0.043 * static_cast< Real >( i ) +
            0.113 * static_cast< Real >( ( i * 5 ) % 17 ) +
            0.019 * static_cast< Real >( ( i * i + 3 * i ) % 23 );
      } );
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyGradGradWithInitial(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & input,
   const Vector & initial )
{
   Vector output( initial );

   auto op =
      MakeGradGradOperator< KernelPolicy >( fe_space, integration_rule );
   auto dofs_in =
      MakeReadOnlyElementTensorView< KernelPolicy >( fe_space, input );
   auto dofs_out =
      MakeReadWriteElementTensorView< KernelPolicy >( fe_space, output );
   op.Apply( dofs_in, dofs_out );
   GENDIL_DEVICE_SYNC;

   return output;
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
         success =
            CheckZeroWorkItems< LegacyConfig >(
               label,
               integer_sentinel ) && success;
      }
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
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   Vector input = MakeInputVector(
      fe_space.GetNumberOfFiniteElementDofs() );
   auto zero =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         0.0 );

   auto y_batch1 =
      ApplyGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batchn =
      ApplyGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   std::cout << label << " L2, num_cells = " << num_cells << '\n';

   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyGradGradWithInitial< LegacyConfig >(
            fe_space,
            integration_rule,
            input,
            zero );
      success =
         CheckScaledL2Close(
            "DeviceBatchN vs LegacyConfig",
            y_batchn,
            y_legacy,
            tolerance ) && success;
      success =
         CheckScaledL2Close(
            "DeviceBatch1 vs LegacyConfig",
            y_batch1,
            y_legacy,
            tolerance ) && success;
   }

   success =
      CheckScaledL2Close(
         "DeviceBatchN vs DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;

   auto sentinel_initial =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         output_sentinel );
   auto direct_sentinel =
      ApplyGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         sentinel_initial );

   success =
      CheckScaledL2Close(
         "DeviceBatchN L2 overwrite from sentinel",
         direct_sentinel,
         y_batchn,
         tolerance ) && success;
   success =
      CheckNoValue(
         "DeviceBatchN L2 overwrite",
         direct_sentinel,
         output_sentinel ) && success;

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
         success =
            CheckZeroWorkItems< LegacyConfig >(
               label,
               integer_sentinel ) && success;
      }
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

   Vector input = MakeInputVector(
      fe_space.GetNumberOfFiniteElementDofs() );
   auto zero =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         0.0 );
   auto baseline =
      MakeBaselineVector( fe_space.GetNumberOfFiniteElementDofs() );

   auto y_batch1 =
      ApplyGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batchn =
      ApplyGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   std::cout << label << " H1, num_cells = " << num_cells << '\n';

   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyGradGradWithInitial< LegacyConfig >(
            fe_space,
            integration_rule,
            input,
            zero );
      success =
         CheckScaledL2Close(
            "DeviceBatchN vs LegacyConfig",
            y_batchn,
            y_legacy,
            tolerance ) && success;
      success =
         CheckScaledL2Close(
            "DeviceBatch1 vs LegacyConfig",
            y_batch1,
            y_legacy,
            tolerance ) && success;
   }

   success =
      CheckScaledL2Close(
         "DeviceBatchN vs DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;

   auto direct_baseline =
      ApplyGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         baseline );
   auto expected_accumulated = AddVectors( baseline, y_batchn );
   success =
      CheckScaledL2Close(
         "DeviceBatchN H1 accumulation from baseline",
         direct_baseline,
         expected_accumulated,
         tolerance ) && success;

   FreeDevicePointer( indices );
   FreeHostPointer( indices );

   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   bool CompareLegacy >
bool Run2DL2GradientSensitivityCase( const char * label )
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

   static constexpr Integer order_x = 3;
   static constexpr Integer order_y = 2;
   static constexpr Integer num_quad_x = order_x + 3;
   static constexpr Integer num_quad_y = order_y + 4;

   const GlobalIndex num_cells_x = 3;
   const GlobalIndex num_cells_y = 5;
   Cartesian1DMesh mesh_x(
      1.0 / static_cast< Real >( num_cells_x ),
      num_cells_x );
   Cartesian1DMesh mesh_y(
      0.7 / static_cast< Real >( num_cells_y ),
      num_cells_y );
   auto mesh = MakeCartesianProductMesh( mesh_x, mesh_y );

   FiniteElementOrders< order_x, order_y > orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_x, num_quad_y > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   Vector input = Make2DInputVector(
      fe_space.GetNumberOfFiniteElementDofs() );
   auto zero =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         0.0 );

   auto y_batch1 =
      ApplyGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batchn =
      ApplyGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   std::cout << label << " 2D L2 gradient sensitivity\n";

   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyGradGradWithInitial< LegacyConfig >(
            fe_space,
            integration_rule,
            input,
            zero );
      success =
         CheckScaledL2Close(
            "DeviceBatchN vs LegacyConfig",
            y_batchn,
            y_legacy,
            tolerance ) && success;
      success =
         CheckScaledL2Close(
            "DeviceBatch1 vs LegacyConfig",
            y_batch1,
            y_legacy,
            tolerance ) && success;
   }

   success =
      CheckScaledL2Close(
         "DeviceBatchN vs DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;
   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   bool CompareLegacy >
bool RunGradGradBatchCases( const char * label )
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

bool TestThreadedGradGrad()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 4 );

   bool success = true;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 1, true >(
         "ThreadBlockLayout<4>, BatchSize=1" ) && success;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 2, true >(
         "ThreadBlockLayout<4>, BatchSize=2" ) && success;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 4, true >(
         "ThreadBlockLayout<4>, BatchSize=4" ) && success;
   success =
      RunGradGradBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         true >(
            "ThreadBlockLayout<4>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestRegisterOnlyGradGrad()
{
   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 0;

   bool success = true;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 1, false >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 2, false >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 4, false >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunGradGradBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         false >(
            "ThreadBlockLayout<>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestIrregularGradGradDiagnostic()
{
   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static_assert( Layout::GetNumberOfThreads() == 15 );

   std::cout
      << "Running diagnostic irregular ThreadBlockLayout<3,5> "
      << "GradGrad cases. For this 1D operator, extra logical thread "
      << "dimensions are expected to be idle. The "
      << "BatchSize=device_warp_size case is isolated in the GradGrad "
      << "debug diagnostics.\n";

   bool success = true;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 1, true >(
         "ThreadBlockLayout<3,5>, BatchSize=1 diagnostic" ) && success;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 2, true >(
         "ThreadBlockLayout<3,5>, BatchSize=2 diagnostic" ) && success;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 4, true >(
         "ThreadBlockLayout<3,5>, BatchSize=4 diagnostic" ) && success;
   return success;
}

bool Test2DL2GradientSensitivity()
{
   bool success = true;

   using RegisterLayout = ThreadBlockLayout<>;
   success =
      Run2DL2GradientSensitivityCase<
         RegisterLayout,
         0,
         2,
         false >(
            "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      Run2DL2GradientSensitivityCase<
         RegisterLayout,
         0,
         4,
         false >(
            "ThreadBlockLayout<>, BatchSize=4" ) && success;

   using ThreadedLayout = ThreadBlockLayout< 4 >;
   static constexpr Integer ThreadedMaxSharedDimensions = 1;
   success =
      Run2DL2GradientSensitivityCase<
         ThreadedLayout,
         ThreadedMaxSharedDimensions,
         2,
         true >(
            "ThreadBlockLayout<4>, BatchSize=2" ) && success;
   success =
      Run2DL2GradientSensitivityCase<
         ThreadedLayout,
         ThreadedMaxSharedDimensions,
         4,
         true >(
            "ThreadBlockLayout<4>, BatchSize=4" ) && success;

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRegisterOnlyGradGrad() && success;
   success = TestThreadedGradGrad() && success;
   success = TestIrregularGradGradDiagnostic() && success;
   success = Test2DL2GradientSensitivity() && success;

   return success ? 0 : 1;
}

#endif
