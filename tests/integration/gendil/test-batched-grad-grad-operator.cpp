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

enum class OutputViewMode
{
   production_write_only_output_view,
   diagnostic_read_write_output_view
};

const char * OutputViewModeName( const OutputViewMode mode )
{
   switch ( mode )
   {
      case OutputViewMode::production_write_only_output_view:
         return "production_write_only_output_view";
      case OutputViewMode::diagnostic_read_write_output_view:
         return "diagnostic_read_write_output_view";
   }
   return "unknown_output_view";
}

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
   OutputViewMode Mode,
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyGradGradWithInitialView(
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
   auto dofs_in =
      MakeReadOnlyElementTensorView< KernelPolicy >( fe_space, input );
   if constexpr ( Mode == OutputViewMode::production_write_only_output_view )
   {
      auto dofs_out =
         MakeWriteOnlyElementTensorView< KernelPolicy >( fe_space, output );
      op.Apply( dofs_in, dofs_out );
   }
   else
   {
      auto dofs_out =
         MakeReadWriteElementTensorView< KernelPolicy >( fe_space, output );
      op.Apply( dofs_in, dofs_out );
   }
   GENDIL_DEVICE_SYNC;

   return output;
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
   return ApplyGradGradWithInitialView<
      OutputViewMode::diagnostic_read_write_output_view,
      KernelPolicy >(
         fe_space,
         integration_rule,
         input,
         initial );
}

Vector SubtractVectors( const Vector & a, const Vector & b )
{
   GENDIL_VERIFY( a.Size() == b.Size(), "Vector sizes do not match." );

   Vector result( a.Size() );
   Real * result_data = result.WriteHostData();
   const Real * a_data = a.ReadHostData();
   const Real * b_data = b.ReadHostData();
   for ( Integer i = 0; i < result.Size(); ++i )
   {
      result_data[ i ] = a_data[ i ] - b_data[ i ];
   }
   return result;
}

bool PrintScaledL2Diagnostic(
   const char * label,
   const Vector & observed,
   const Vector & expected,
   const Real tolerance,
   const bool gating )
{
   const Real abs_error = AbsoluteL2Error( observed, expected );
   const Real scale = std::max( Real{ 1.0 }, L2Norm( expected ) );
   const Real scaled_error = abs_error / scale;
   const bool ok =
      std::isfinite( abs_error ) &&
      std::isfinite( scaled_error ) &&
      scaled_error <= tolerance;

   std::cout << label << " scaled L2 error = " << scaled_error << '\n';
   if ( gating && !ok )
   {
      std::cout << "FAILED: " << label << " exceeded tolerance "
                << tolerance << ".\n";
   }
   return ok;
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

void PrintFirstMismatches(
   const char * label,
   const char * output_view_mode,
   const char * batch_classification,
   const Vector & final,
   const Vector & initial,
   const Vector & oracle,
   const Real tolerance )
{
   GENDIL_VERIFY( final.Size() == initial.Size(), "Vector sizes do not match." );
   GENDIL_VERIFY( final.Size() == oracle.Size(), "Vector sizes do not match." );

   const Real * final_data = final.ReadHostData();
   const Real * initial_data = initial.ReadHostData();
   const Real * oracle_data = oracle.ReadHostData();
   Integer printed = 0;
   for ( Integer i = 0; i < final.Size(); ++i )
   {
      const Real scale = std::max( Real{ 1.0 }, std::abs( oracle_data[ i ] ) );
      if ( std::abs( final_data[ i ] - oracle_data[ i ] ) > tolerance * scale )
      {
         std::cout << label << " first mismatch: output-view="
                   << output_view_mode
                   << ", batch=" << batch_classification
                   << ", index=" << i
                   << ", oracle value=" << oracle_data[ i ]
                   << ", initialized value=" << initial_data[ i ]
                   << ", final value=" << final_data[ i ]
                   << ", final-minus-initial value="
                   << final_data[ i ] - initial_data[ i ] << ".\n";
         ++printed;
         if ( printed == 8 )
         {
            return;
         }
      }
   }
}

bool CheckScaledL2CloseWithMismatches(
   const char * label,
   const char * output_view_mode,
   const char * batch_classification,
   const Vector & final,
   const Vector & initial,
   const Vector & oracle,
   const Real tolerance )
{
   const bool ok =
      CheckScaledL2Close(
         label,
         final,
         oracle,
         tolerance );
   if ( !ok )
   {
      PrintFirstMismatches(
         label,
         output_view_mode,
         batch_classification,
         final,
         initial,
         oracle,
         tolerance );
   }
   return ok;
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
      ApplyGradGradWithInitialView<
         OutputViewMode::production_write_only_output_view,
         DeviceBatch1 >(
            fe_space,
            integration_rule,
            input,
            zero );
   auto y_batchn =
      ApplyGradGradWithInitialView<
         OutputViewMode::production_write_only_output_view,
         DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   const char * output_view_mode =
      OutputViewModeName( OutputViewMode::production_write_only_output_view );
   const char * batch_classification =
      BatchClassification< BatchSize >( num_cells );
   std::cout << label << " L2, num_cells = " << num_cells
             << ", output-view=" << output_view_mode
             << ", batch=" << batch_classification << '\n';

   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyGradGradWithInitialView<
            OutputViewMode::production_write_only_output_view,
            LegacyConfig >(
               fe_space,
               integration_rule,
               input,
               zero );
      success =
         CheckScaledL2CloseWithMismatches(
            "DeviceBatchN vs LegacyConfig",
            output_view_mode,
            batch_classification,
            y_batchn,
            zero,
            y_legacy,
            tolerance ) && success;
      success =
         CheckScaledL2CloseWithMismatches(
            "DeviceBatch1 vs LegacyConfig",
            output_view_mode,
            batch_classification,
            y_batch1,
            zero,
            y_legacy,
            tolerance ) && success;
   }

   success =
      CheckScaledL2CloseWithMismatches(
         "DeviceBatchN vs DeviceBatch1",
         output_view_mode,
         batch_classification,
         y_batchn,
         zero,
         y_batch1,
         tolerance ) && success;

   auto sentinel_initial =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         output_sentinel );
   auto direct_sentinel =
      ApplyGradGradWithInitialView<
         OutputViewMode::production_write_only_output_view,
         DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            sentinel_initial );

   const bool sentinel_overwrite_ok =
      CheckScaledL2CloseWithMismatches(
         "DeviceBatchN L2 overwrite from sentinel",
         output_view_mode,
         batch_classification,
         direct_sentinel,
         sentinel_initial,
         y_batchn,
         tolerance );
   success = sentinel_overwrite_ok && success;
   success =
      CheckNoValue(
         "DeviceBatchN L2 overwrite",
         direct_sentinel,
         output_sentinel ) && success;

   return success;
}

template < OutputViewMode Mode >
bool RunFocusedL2OutputViewDiagnostic(
   const GlobalIndex num_cells,
   const bool production_write_only_passed = false )
{
   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static constexpr Integer BatchSize = device_warp_size;
   using DeviceBatch1 =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchN =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   const char * mode_label = OutputViewModeName( Mode );
   const char * batch_classification =
      BatchClassification< BatchSize >( num_cells );
   if ( !LaunchConfigurationFits< Layout, BatchSize >( mode_label ) )
   {
      return true;
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
   auto sentinel_initial =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         output_sentinel );
   auto baseline_initial =
      MakeBaselineVector( fe_space.GetNumberOfFiniteElementDofs() );

   auto oracle =
      ApplyGradGradWithInitialView<
         OutputViewMode::production_write_only_output_view,
         DeviceBatch1 >(
            fe_space,
            integration_rule,
            input,
            zero );
   auto y_zero =
      ApplyGradGradWithInitialView< Mode, DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_sentinel =
      ApplyGradGradWithInitialView< Mode, DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         sentinel_initial );
   auto y_baseline =
      ApplyGradGradWithInitialView< Mode, DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         baseline_initial );

   auto y_sentinel_minus_initial =
      SubtractVectors( y_sentinel, sentinel_initial );
   auto y_baseline_minus_initial =
      SubtractVectors( y_baseline, baseline_initial );

   constexpr Real tolerance = 1.0e-10;
   constexpr bool production_mode =
      Mode == OutputViewMode::production_write_only_output_view;
   bool success = true;

   std::cout
      << "Focused GradGrad L2 output-view diagnostic: " << mode_label
      << ", ThreadBlockLayout<5>, BatchSize=device_warp_size, num_cells="
      << num_cells << ", batch=" << batch_classification << ".\n";
   std::cout
      << "  Audit: production GradGrad operator()/Mult() uses a write-only "
      << "output view; diagnostic_read_write_output_view is diagnostic only. "
      << "Scalar L2 ThreadedWriteDofs<Add=false> should assign, while H1 "
      << "shared DOFs accumulate. No GradGrad local staging buffer is expected "
      << "to initialize from global y.\n";
   if ( IsFinalPartialBatch< BatchSize >( num_cells ) )
   {
      std::cout
         << "  Audit: this is a final partial batch. "
         << "production_write_only_output_view is a gating production "
         << "validation path; diagnostic_read_write_output_view is "
         << "non-gating classification only.\n";
   }

   const bool zero_ok =
      PrintScaledL2Diagnostic(
         "focused y_zero vs oracle",
         y_zero,
         oracle,
         tolerance,
         production_mode );
   const bool sentinel_ok =
      PrintScaledL2Diagnostic(
         "focused y_sentinel vs oracle",
         y_sentinel,
         oracle,
         tolerance,
         production_mode );
   const bool sentinel_delta_ok =
      PrintScaledL2Diagnostic(
         "focused y_sentinel - sentinel vs oracle",
         y_sentinel_minus_initial,
         oracle,
         tolerance,
         false );
   const bool baseline_ok =
      PrintScaledL2Diagnostic(
         "focused y_baseline vs oracle",
         y_baseline,
         oracle,
         tolerance,
         production_mode );
   const bool baseline_delta_ok =
      PrintScaledL2Diagnostic(
         "focused y_baseline - baseline vs oracle",
         y_baseline_minus_initial,
         oracle,
         tolerance,
         false );

   if ( !zero_ok )
   {
      PrintFirstMismatches(
         "focused y_zero vs oracle",
         mode_label,
         batch_classification,
         y_zero,
         zero,
         oracle,
         tolerance );
   }
   if ( !sentinel_ok )
   {
      PrintFirstMismatches(
         "focused y_sentinel vs oracle",
         mode_label,
         batch_classification,
         y_sentinel,
         sentinel_initial,
         oracle,
         tolerance );
   }
   if ( !baseline_ok )
   {
      PrintFirstMismatches(
         "focused y_baseline vs oracle",
         mode_label,
         batch_classification,
         y_baseline,
         baseline_initial,
         oracle,
         tolerance );
   }

   if ( zero_ok && sentinel_ok && baseline_ok )
   {
      std::cout << "  CLASSIFICATION: " << mode_label
                << " overwrites the initial output state for this case.\n";
   }
   else if ( sentinel_delta_ok && baseline_delta_ok )
   {
      if constexpr ( production_mode )
      {
         std::cout
            << "  CLASSIFICATION: production output-view/write contract "
            << "violation; subtracting the initial output recovers the oracle.\n";
      }
      else
      {
         if ( production_write_only_passed )
         {
            std::cout
               << "  CLASSIFICATION: diagnostic_output_view_artifact "
               << "(diagnostic_read_write_additive_artifact); "
               << "production_write_only_output_view passed for this "
               << batch_classification
               << ", so the read-write diagnostic does not narrow the "
               << "production GradGrad validation contract.\n";
         }
         else
         {
            std::cout
               << "  CLASSIFICATION: diagnostic_read_write_additive_artifact; "
               << "subtracting the initial output recovers the oracle.\n";
         }
      }
   }
   else if ( zero_ok )
   {
      if constexpr ( production_mode )
      {
         std::cout
            << "  CLASSIFICATION: output-state dependent mismatch; "
            << "subtracting the initial output does not fully recover the "
            << "oracle.\n";
      }
      else if ( production_write_only_passed )
      {
         std::cout
            << "  CLASSIFICATION: diagnostic_output_view_artifact; "
            << "production_write_only_output_view passed for this "
            << batch_classification
            << ", so the read-write diagnostic does not narrow the "
            << "production GradGrad validation contract.\n";
      }
      else
      {
         std::cout
            << "  CLASSIFICATION: diagnostic read-write output-state "
            << "dependent mismatch; subtracting the initial output does not "
            << "fully recover the oracle.\n";
      }
   }
   else
   {
      if constexpr ( production_mode )
      {
         std::cout
            << "  CLASSIFICATION: zero-initialized production output also "
            << "mismatches the oracle, suggesting a full GradGrad staging or "
            << "batching issue.\n";
      }
      else if ( production_write_only_passed )
      {
         std::cout
            << "  CLASSIFICATION: diagnostic_output_view_artifact; "
            << "production_write_only_output_view passed for this "
            << batch_classification
            << ", so the read-write diagnostic does not narrow the "
            << "production GradGrad validation contract.\n";
      }
      else
      {
         std::cout
            << "  CLASSIFICATION: zero-initialized diagnostic read-write "
            << "output also mismatches the oracle.\n";
      }
   }

   if constexpr ( production_mode )
   {
      success = zero_ok && sentinel_ok && baseline_ok;
   }
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
   const char * output_view_mode = "production_h1_accumulation_output_view";
   const char * batch_classification =
      BatchClassification< BatchSize >( num_cells );
   std::cout << label << " H1, num_cells = " << num_cells
             << ", output-view=" << output_view_mode
             << ", batch=" << batch_classification << '\n';

   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyGradGradWithInitial< LegacyConfig >(
            fe_space,
            integration_rule,
            input,
            zero );
      success =
         CheckScaledL2CloseWithMismatches(
            "DeviceBatchN vs LegacyConfig",
            output_view_mode,
            batch_classification,
            y_batchn,
            zero,
            y_legacy,
            tolerance ) && success;
      success =
         CheckScaledL2CloseWithMismatches(
            "DeviceBatch1 vs LegacyConfig",
            output_view_mode,
            batch_classification,
            y_batch1,
            zero,
            y_legacy,
            tolerance ) && success;
   }

   success =
      CheckScaledL2CloseWithMismatches(
         "DeviceBatchN vs DeviceBatch1",
         output_view_mode,
         batch_classification,
         y_batchn,
         zero,
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
      CheckScaledL2CloseWithMismatches(
         "DeviceBatchN H1 accumulation from baseline",
         output_view_mode,
         batch_classification,
         direct_baseline,
         baseline,
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
      ApplyGradGradWithInitialView<
         OutputViewMode::production_write_only_output_view,
         DeviceBatch1 >(
            fe_space,
            integration_rule,
            input,
            zero );
   auto y_batchn =
      ApplyGradGradWithInitialView<
         OutputViewMode::production_write_only_output_view,
         DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   const char * output_view_mode =
      OutputViewModeName( OutputViewMode::production_write_only_output_view );
   const char * batch_classification =
      BatchClassification< BatchSize >( num_cells );
   std::cout
      << label
      << " 2D L2 gradient-component sensitivity, output-view="
      << output_view_mode << ", batch=" << batch_classification
      << "; this is not a 2D logical-thread-layout coverage test.\n";

   if constexpr ( CompareLegacy )
   {
      auto y_legacy =
         ApplyGradGradWithInitialView<
            OutputViewMode::production_write_only_output_view,
            LegacyConfig >(
               fe_space,
               integration_rule,
               input,
               zero );
      success =
         CheckScaledL2CloseWithMismatches(
            "DeviceBatchN vs LegacyConfig",
            output_view_mode,
            batch_classification,
            y_batchn,
            zero,
            y_legacy,
            tolerance ) && success;
      success =
         CheckScaledL2CloseWithMismatches(
            "DeviceBatch1 vs LegacyConfig",
            output_view_mode,
            batch_classification,
            y_batch1,
            zero,
            y_legacy,
            tolerance ) && success;
   }

   success =
      CheckScaledL2CloseWithMismatches(
         "DeviceBatchN vs DeviceBatch1",
         output_view_mode,
         batch_classification,
         y_batchn,
         zero,
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
   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 5 );

   std::cout
      << "Threaded GradGrad validation scope: production validation covers "
      << "full and final partial batches. Diagnostic read-write output-view "
      << "experiments are non-gating classification tools only.\n";

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

bool TestFocusedL2OutputViewDiagnostic()
{
   bool success = true;
   const bool production_64_success =
      RunFocusedL2OutputViewDiagnostic<
         OutputViewMode::production_write_only_output_view >( 64 );
   success = production_64_success && success;
   success =
      RunFocusedL2OutputViewDiagnostic<
         OutputViewMode::diagnostic_read_write_output_view >(
            64,
            production_64_success ) &&
      success;
   const bool production_65_success =
      RunFocusedL2OutputViewDiagnostic<
         OutputViewMode::production_write_only_output_view >( 65 );
   success = production_65_success && success;
   success =
      RunFocusedL2OutputViewDiagnostic<
         OutputViewMode::diagnostic_read_write_output_view >(
            65,
            production_65_success ) &&
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
   std::cout
      << "Skipping ThreadBlockLayout<3,5> GradGrad diagnostic: the current "
      << "threaded helper contract requires the mapped 1D thread dimension "
      << "to cover the local DOF/quadrature extent.\n";
   return true;
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
   success = TestFocusedL2OutputViewDiagnostic() && success;
   success = TestIrregularGradGradDiagnostic() && success;
   success = Test2DL2GradientSensitivity() && success;

   return success ? 0 : 1;
}

#endif
