// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <cmath>
#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-grad-grad-debug-variants skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{
template < typename A, typename B >
Real ScaledL2Error( const A & observed, const B & expected )
{
   const Real abs_error = AbsoluteL2Error( observed, expected );
   const Real scale = std::max( Real{ 1.0 }, L2Norm( expected ) );
   return abs_error / scale;
}

template < typename A, typename B >
bool PrintScaledL2Diagnostic(
   const char * label,
   const A & observed,
   const B & expected,
   const Real tolerance )
{
   const Real scaled_error = ScaledL2Error( observed, expected );
   std::cout << label << " scaled L2 error = " << scaled_error << '\n';
   return std::isfinite( scaled_error ) && scaled_error <= tolerance;
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

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyFilteredGradGradWithInitial(
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

template <
   typename IntegrationRule,
   typename KernelContext,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData,
   typename DofsInView,
   typename DofsOutView >
GENDIL_HOST_DEVICE
void DebugGradGradElementOperator(
   KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out,
   const bool write_output )
{
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );
   auto Gu = InterpolateGradient( kernel_conf, element_quad_data, u );
   auto GDGu =
      ApplyGradientTestFunctionsAtQPoints(
         kernel_conf,
         element_quad_data,
         Gu );
   auto BGDGu = ApplyTestFunctions( kernel_conf, element_quad_data, GDGu );

   if ( write_output )
   {
      WriteDofs( kernel_conf, fe_space, element_index, BGDGu, dofs_out );
   }
}

template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData,
   typename DofsInView,
   typename DofsOutView >
void DebugAllCandidateGradGradExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   const GlobalIndex num_cells = fe_space.GetNumberOfCells();
   KernelConfiguration::BlockLoop(
      num_cells,
      [=] GENDIL_HOST_DEVICE ( const KernelConfiguration & kernel ) mutable
      {
         const bool active = kernel.IsActive( num_cells );
         const GlobalIndex element_index =
            active ? kernel.WorkItemIndex() : GlobalIndex( 0 );

         constexpr size_t required_shared_mem =
            required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[
            KernelContext<
               KernelConfiguration,
               required_shared_mem >::shared_memory_block_size ];

         KernelContext< KernelConfiguration, required_shared_mem >
            kernel_conf( _shared_mem, kernel );

         DebugGradGradElementOperator< IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            dofs_in,
            dofs_out,
            active );
      } );
}

template <
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData,
   typename DofsInView,
   typename DofsOutView >
void DebugEarlyReturnGradGradExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out )
{
   const GlobalIndex num_cells = fe_space.GetNumberOfCells();
   KernelConfiguration::BlockLoop(
      num_cells,
      [=] GENDIL_HOST_DEVICE ( const KernelConfiguration & kernel ) mutable
      {
         // Test-only MFEM-style control-flow diagnostic. This differs from the
         // current CellIterator wrapper shape by making the inactive-candidate
         // branch an explicit early return inside the kernel body. A passing
         // result here is only a codegen/control-flow clue; it is not a
         // synchronization contract for bodies with block-wide Sync().
         if ( !kernel.IsActive( num_cells ) )
         {
            return;
         }

         const GlobalIndex element_index = kernel.WorkItemIndex();

         constexpr size_t required_shared_mem =
            required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[
            KernelContext<
               KernelConfiguration,
               required_shared_mem >::shared_memory_block_size ];

         KernelContext< KernelConfiguration, required_shared_mem >
            kernel_conf( _shared_mem, kernel );

         DebugGradGradElementOperator< IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            dofs_in,
            dofs_out,
            true );
      } );
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyAllCandidateGradGradWithInitial(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & input,
   const Vector & initial )
{
   Vector output( initial );

   auto dofs_in =
      MakeReadOnlyElementTensorView< KernelPolicy >( fe_space, input );
   auto dofs_out =
      MakeReadWriteElementTensorView< KernelPolicy >( fe_space, output );

   using Mesh = typename FiniteElementSpace::mesh_type;
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;

   typename Mesh::cell_type::template QuadData< Rule > mesh_quad_data{};
   auto element_quad_data =
      MakeDofToQuad< ShapeFunctions, Rule >();

   (void) integration_rule;

   DebugAllCandidateGradGradExplicitOperator< KernelPolicy, Rule >(
      fe_space,
      mesh_quad_data,
      element_quad_data,
      dofs_in,
      dofs_out );
   GENDIL_DEVICE_SYNC;

   return output;
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyEarlyReturnGradGradWithInitial(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & input,
   const Vector & initial )
{
   Vector output( initial );

   auto dofs_in =
      MakeReadOnlyElementTensorView< KernelPolicy >( fe_space, input );
   auto dofs_out =
      MakeReadWriteElementTensorView< KernelPolicy >( fe_space, output );

   using Mesh = typename FiniteElementSpace::mesh_type;
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;

   typename Mesh::cell_type::template QuadData< Rule > mesh_quad_data{};
   auto element_quad_data =
      MakeDofToQuad< ShapeFunctions, Rule >();

   (void) integration_rule;

   DebugEarlyReturnGradGradExplicitOperator< KernelPolicy, Rule >(
      fe_space,
      mesh_quad_data,
      element_quad_data,
      dofs_in,
      dofs_out );
   GENDIL_DEVICE_SYNC;

   return output;
}

template < GlobalIndex NumCells >
bool RunDebugVariantCase()
{
   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static constexpr Integer BatchSize = device_warp_size;

   if ( !LaunchConfigurationFits< Layout, BatchSize >(
           "ThreadBlockLayout<3,5>, BatchSize=device_warp_size" ) )
   {
      return true;
   }

   using LegacyConfig =
      ThreadFirstKernelConfiguration< Layout, MaxSharedDimensions >;
   using DeviceBatch1 =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchN =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   static constexpr Integer order = 3;
   static constexpr Integer num_quad_1d = order + 2;

   const Real h = 1.0 / static_cast< Real >( NumCells );
   Cartesian1DMesh mesh( h, NumCells );

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
      ApplyFilteredGradGradWithInitial< LegacyConfig >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batch1 =
      ApplyFilteredGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_filtered =
      ApplyFilteredGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_early_return =
      ApplyEarlyReturnGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_all_candidate =
      ApplyAllCandidateGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   std::cout
      << "ThreadBlockLayout<3,5>, BatchSize=device_warp_size debug, "
      << "num_cells = " << NumCells << '\n';

   success =
      CheckScaledL2Close(
         "DeviceBatch1 vs LegacyConfig",
         y_batch1,
         y_legacy,
         tolerance ) && success;
   const bool filtered_matches_legacy =
      PrintScaledL2Diagnostic(
         "Filtered DeviceBatchN vs LegacyConfig",
         y_filtered,
         y_legacy,
         tolerance );
   const bool filtered_matches_batch1 =
      PrintScaledL2Diagnostic(
         "Filtered DeviceBatchN vs DeviceBatch1",
         y_filtered,
         y_batch1,
         tolerance );
   const bool early_return_matches_legacy =
      PrintScaledL2Diagnostic(
         "Early-return DeviceBatchN vs LegacyConfig",
         y_early_return,
         y_legacy,
         tolerance );
   const bool early_return_matches_batch1 =
      PrintScaledL2Diagnostic(
         "Early-return DeviceBatchN vs DeviceBatch1",
         y_early_return,
         y_batch1,
         tolerance );
   success =
      CheckScaledL2Close(
         "All-candidate DeviceBatchN vs LegacyConfig",
         y_all_candidate,
         y_legacy,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "All-candidate DeviceBatchN vs DeviceBatch1",
         y_all_candidate,
         y_batch1,
         tolerance ) && success;

   if constexpr ( NumCells % BatchSize == 0 )
   {
      if ( !filtered_matches_legacy || !filtered_matches_batch1 )
      {
         std::cout
            << "FAILED: filtered production-style GradGrad mismatched for "
            << "a full batch with no inactive candidates.\n";
         success = false;
      }
      if ( !early_return_matches_legacy || !early_return_matches_batch1 )
      {
         std::cout
            << "FAILED: early-return GradGrad mismatched for a full batch "
            << "with no inactive candidates.\n";
         success = false;
      }
   }
   else if ( !filtered_matches_legacy || !filtered_matches_batch1 )
   {
      std::cout
         << "DIAGNOSTIC: filtered production-style GradGrad mismatched only "
         << "in the final partial batch, while the test-only all-candidate "
         << "guarded-write variant matched. This supports the filtered "
         << "CellIterator plus block-wide Sync() hypothesis.\n";
      if ( early_return_matches_legacy && early_return_matches_batch1 )
      {
         std::cout
            << "DIAGNOSTIC: the MFEM-style early-return variant matched "
            << "where filtered invocation failed. Treat this as a "
            << "codegen/control-flow distinction only, not as proof that "
            << "early return is a robust synchronization contract.\n";
      }
      else
      {
         std::cout
            << "DIAGNOSTIC: the MFEM-style early-return variant also "
            << "mismatched. The robust passing diagnostic remains the "
            << "all-candidate guarded-write variant.\n";
      }
   }
   else
   {
      std::cout
         << "DIAGNOSTIC: filtered production-style GradGrad matched in the "
         << "final partial batch; the filtered-sync hazard was not "
         << "reproduced in this run.\n";
   }

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = RunDebugVariantCase< 64 >() && success;
   success = RunDebugVariantCase< 65 >() && success;
   return success ? 0 : 1;
}

#endif
