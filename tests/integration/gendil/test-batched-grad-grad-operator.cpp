// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
Vector ApplyProductionGradGradWithInitial(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & input,
   const Vector & initial )
{
   GENDIL_DEVICE_SYNC;
   Vector output( initial );
   GENDIL_DEVICE_SYNC;

   auto op =
      MakeGradGradOperator< KernelPolicy >( fe_space, integration_rule );
   op( input, output );
   GENDIL_DEVICE_SYNC;

   return output;
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyGradGradWithDevicePoisonedOutput(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & input,
   const Real poison_value )
{
   Vector output( fe_space.GetNumberOfFiniteElementDofs() );
   Real * output_data = output.WriteDeviceData();
   const size_t output_size = output.Size();
   DeviceLoop(
      output_size,
      [=] GENDIL_HOST_DEVICE ( Integer i )
      {
         output_data[ i ] = poison_value;
      } );
   GENDIL_DEVICE_SYNC;

   auto op =
      MakeGradGradOperator< KernelPolicy >( fe_space, integration_rule );
   op( input, output );
   GENDIL_DEVICE_SYNC;

   return output;
}

struct ErrorSummary
{
   Real scaled_l2_error;
   Real max_abs_error;
   bool ok;
};

ErrorSummary ComputeErrorSummary(
   const Vector & observed,
   const Vector & expected,
   const Real tolerance )
{
   GENDIL_VERIFY(
      observed.Size() == expected.Size(),
      "Vector sizes do not match." );

   const Real * observed_data = observed.ReadHostData();
   const Real * expected_data = expected.ReadHostData();
   Real err_sq = 0.0;
   Real expected_norm_sq = 0.0;
   Real max_abs_error = 0.0;

   for ( Integer i = 0; i < observed.Size(); ++i )
   {
      const Real diff = observed_data[ i ] - expected_data[ i ];
      err_sq += diff * diff;
      expected_norm_sq += expected_data[ i ] * expected_data[ i ];
      max_abs_error = std::max( max_abs_error, std::abs( diff ) );
   }

   const Real abs_error = std::sqrt( err_sq );
   const Real scale = std::max( Real{ 1.0 }, std::sqrt( expected_norm_sq ) );
   const Real scaled_l2_error = abs_error / scale;
   const bool ok =
      std::isfinite( max_abs_error ) &&
      std::isfinite( scaled_l2_error ) &&
      scaled_l2_error <= tolerance;

   return { scaled_l2_error, max_abs_error, ok };
}

template < Integer BatchSize >
bool IsFinalPartialBatch( const GlobalIndex num_cells )
{
   return BatchSize > 1 && num_cells > 0 && num_cells % BatchSize != 0;
}

template < Integer BatchSize >
const char * BatchClassification( const GlobalIndex num_cells )
{
   return IsFinalPartialBatch< BatchSize >( num_cells )
      ? "partial-batch"
      : "full-batch";
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

void PrintProductionResult(
   const char * label,
   const GlobalIndex num_cells,
   const char * batch_classification,
   const char * space,
   const bool has_legacy,
   const ErrorSummary & legacy_summary,
   const ErrorSummary & batch1_summary,
   const Real tolerance )
{
   const bool pass =
      batch1_summary.ok && ( !has_legacy || legacy_summary.ok );

   std::cout
      << "PRODUCTION GradGrad: " << label
      << ", num_cells=" << num_cells
      << ", batch=" << batch_classification
      << ", space=" << space
      << ", result=" << ( pass ? "PASS" : "FAIL" );

   if ( has_legacy )
   {
      std::cout
         << ", vs LegacyConfig scaled_l2="
         << legacy_summary.scaled_l2_error
         << ", max_abs=" << legacy_summary.max_abs_error;
   }
   else
   {
      std::cout << ", vs LegacyConfig=N/A";
   }

   std::cout
      << ", vs same-layout DeviceBatch1 scaled_l2="
      << batch1_summary.scaled_l2_error
      << ", max_abs=" << batch1_summary.max_abs_error << ".\n";

   if ( !pass )
   {
      std::cout
         << "FAILED: production GradGrad " << label
         << ", space=" << space
         << ", num_cells=" << num_cells
         << " exceeded tolerance " << tolerance << ".\n";
   }
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
      ApplyProductionGradGradWithInitial< DeviceBatch1 >(
            fe_space,
            integration_rule,
            input,
            zero );
   auto y_batchn =
      ApplyProductionGradGradWithInitial< DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   const char * batch_classification =
      BatchClassification< BatchSize >( num_cells );

   ErrorSummary legacy_summary{ 0.0, 0.0, true };
   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyProductionGradGradWithInitial< LegacyConfig >(
            fe_space,
            integration_rule,
            input,
            zero );
      legacy_summary =
         ComputeErrorSummary( y_batchn, y_legacy, tolerance );
      success = legacy_summary.ok && success;
   }

   const ErrorSummary batch1_summary =
      ComputeErrorSummary( y_batchn, y_batch1, tolerance );
   success = batch1_summary.ok && success;

   PrintProductionResult(
      label,
      num_cells,
      batch_classification,
      "L2/DG",
      CompareLegacy,
      legacy_summary,
      batch1_summary,
      tolerance );

   return success;
}

bool TestFocusedDeviceResidentOverwriteCoverage()
{
   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static constexpr Integer BatchSize = device_warp_size;
   using LegacyConfig =
      ThreadFirstKernelConfiguration< Layout, MaxSharedDimensions >;
   using DeviceBatch1 =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchN =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   constexpr const char * label =
      "ThreadBlockLayout<5>, BatchSize=device_warp_size "
      "device-resident overwrite coverage";
   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return true;
   }

   static constexpr GlobalIndex num_cells = 64;
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

   auto y_legacy =
      ApplyProductionGradGradWithInitial< LegacyConfig >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batch1 =
      ApplyProductionGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto device_poisoned_output =
      ApplyGradGradWithDevicePoisonedOutput< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         output_sentinel );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   const char * batch_classification =
      BatchClassification< BatchSize >( num_cells );
   const ErrorSummary legacy_summary =
      ComputeErrorSummary( device_poisoned_output, y_legacy, tolerance );
   const ErrorSummary batch1_summary =
      ComputeErrorSummary( device_poisoned_output, y_batch1, tolerance );
   success = legacy_summary.ok && batch1_summary.ok && success;
   success =
      CheckNoValue(
         "Device-resident GradGrad overwrite coverage",
         device_poisoned_output,
         output_sentinel ) && success;

   PrintProductionResult(
      label,
      num_cells,
      batch_classification,
      "L2/DG",
      true,
      legacy_summary,
      batch1_summary,
      tolerance );

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

   auto y_batch1 =
      ApplyProductionGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batchn =
      ApplyProductionGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   const char * batch_classification =
      BatchClassification< BatchSize >( num_cells );

   ErrorSummary legacy_summary{ 0.0, 0.0, true };
   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyProductionGradGradWithInitial< LegacyConfig >(
            fe_space,
            integration_rule,
            input,
            zero );
      legacy_summary =
         ComputeErrorSummary( y_batchn, y_legacy, tolerance );
      success = legacy_summary.ok && success;
   }

   const ErrorSummary batch1_summary =
      ComputeErrorSummary( y_batchn, y_batch1, tolerance );
   success = batch1_summary.ok && success;

   PrintProductionResult(
      label,
      num_cells,
      batch_classification,
      "H1/CG",
      CompareLegacy,
      legacy_summary,
      batch1_summary,
      tolerance );

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
   const GlobalIndex num_cells = num_cells_x * num_cells_y;
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
      ApplyProductionGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batchn =
      ApplyProductionGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   const char * batch_classification =
      BatchClassification< BatchSize >( num_cells );

   ErrorSummary legacy_summary{ 0.0, 0.0, true };
   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyProductionGradGradWithInitial< LegacyConfig >(
            fe_space,
            integration_rule,
            input,
            zero );
      legacy_summary =
         ComputeErrorSummary( y_batchn, y_legacy, tolerance );
      success = legacy_summary.ok && success;
   }

   const ErrorSummary batch1_summary =
      ComputeErrorSummary( y_batchn, y_batch1, tolerance );
   success = batch1_summary.ok && success;

   PrintProductionResult(
      label,
      num_cells,
      batch_classification,
      "2D L2/DG gradient-component sensitivity "
      "(not 2D logical-thread-layout coverage)",
      CompareLegacy,
      legacy_summary,
      batch1_summary,
      tolerance );

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
   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 5 );

   std::cout
      << "Threaded GradGrad production validation scope: full and final "
      << "partial batches are compared against LegacyConfig when available "
      << "and same-layout DeviceBatch1.\n";

   bool success = true;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 1, true >(
         "ThreadBlockLayout<5>, BatchSize=1" ) && success;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 2, true >(
         "ThreadBlockLayout<5>, BatchSize=2" ) && success;
   success =
      RunGradGradBatchCases< Layout, MaxSharedDimensions, 4, true >(
         "ThreadBlockLayout<5>, BatchSize=4" ) && success;
   success =
      RunGradGradBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         true >(
            "ThreadBlockLayout<5>, BatchSize=device_warp_size" ) &&
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

   using ThreadedLayout = ThreadBlockLayout< 6 >;
   static constexpr Integer ThreadedMaxSharedDimensions = 1;
   success =
      Run2DL2GradientSensitivityCase<
         ThreadedLayout,
         ThreadedMaxSharedDimensions,
         2,
         true >(
            "ThreadBlockLayout<6>, BatchSize=2" ) && success;
   success =
      Run2DL2GradientSensitivityCase<
         ThreadedLayout,
         ThreadedMaxSharedDimensions,
         4,
         true >(
            "ThreadBlockLayout<6>, BatchSize=4" ) && success;

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRegisterOnlyGradGrad() && success;
   success = TestThreadedGradGrad() && success;
   success = TestFocusedDeviceResidentOverwriteCoverage() && success;
   success = Test2DL2GradientSensitivity() && success;

   return success ? 0 : 1;
}

#endif
