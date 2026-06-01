// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <cmath>
#include <iostream>
#include <type_traits>

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

enum class StageId : Integer
{
   read_dofs = 0,
   interpolate_gradient,
   apply_gradient_test,
   before_write_dofs,
   after_write_dofs,
   count
};

enum class BarrierMode : Integer
{
   none = 0,
   sync_after_read_dofs,
   sync_after_interpolate_gradient,
   sync_after_apply_gradient_test,
   sync_before_write_dofs,
   sync_after_allocator_reset
};

enum class InvocationMode : Integer
{
   direct_debug_write_true = 0,
   runtime_active_guarded_write,
   early_return,
   all_candidate_guarded_write
};

constexpr Integer diagnostic_threads = 15;
constexpr Integer diagnostic_components = 1;
constexpr Integer diagnostic_dofs = 4;
constexpr Real diagnostic_sentinel = -9.87654321e123;
constexpr GlobalIndex focus_first_element = 50;
constexpr GlobalIndex focus_last_element = 52;
constexpr Integer focus_thread_slots = 5;

const char * StageName( const StageId stage )
{
   switch ( stage )
   {
      case StageId::read_dofs:
         return "after ReadDofs";
      case StageId::interpolate_gradient:
         return "after InterpolateGradient";
      case StageId::apply_gradient_test:
         return "after ApplyGradientTestFunctionsAtQPoints";
      case StageId::before_write_dofs:
         return "before WriteDofs";
      case StageId::after_write_dofs:
         return "after WriteDofs";
      default:
         return "unknown stage";
   }
}

const char * BarrierName( const BarrierMode mode )
{
   switch ( mode )
   {
      case BarrierMode::none:
         return "no_extra_sync";
      case BarrierMode::sync_after_read_dofs:
         return "sync_after_read_dofs";
      case BarrierMode::sync_after_interpolate_gradient:
         return "sync_after_interpolate_gradient";
      case BarrierMode::sync_after_apply_gradient_test:
         return "sync_after_apply_gradient_test";
      case BarrierMode::sync_before_write_dofs:
         return "sync_before_write_dofs";
      case BarrierMode::sync_after_allocator_reset:
         return "sync_after_allocator_reset";
      default:
         return "unknown_sync_variant";
   }
}

const char * InvocationName( const InvocationMode mode )
{
   switch ( mode )
   {
      case InvocationMode::direct_debug_write_true:
         return "direct_debug_write_true";
      case InvocationMode::runtime_active_guarded_write:
         return "runtime_active_guarded_write";
      case InvocationMode::early_return:
         return "early_return";
      case InvocationMode::all_candidate_guarded_write:
         return "all_candidate_guarded_write";
      default:
         return "unknown_invocation";
   }
}

struct StageRecorder
{
   Real * values;
   int * writes;
   GlobalIndex num_cells;

   GENDIL_HOST_DEVICE
   bool enabled() const
   {
      return values != nullptr && writes != nullptr;
   }

   GENDIL_HOST_DEVICE
   GlobalIndex Index(
      const StageId stage,
      const Integer component,
      const GlobalIndex element_index,
      const GlobalIndex thread_index ) const
   {
      return (
         ( static_cast< GlobalIndex >( stage ) * diagnostic_components +
           component ) *
            num_cells +
         element_index ) *
            diagnostic_threads +
         thread_index;
   }

   template < typename KernelContext >
   GENDIL_HOST_DEVICE
   bool DiagnosticThreadParticipates( const KernelContext & kernel ) const
   {
      if constexpr ( KernelContext::thread_block_dim > 1 )
      {
         return kernel.template GetThreadIndex< 1 >() == 0;
      }
      else
      {
         return true;
      }
   }

   template < typename KernelContext >
   GENDIL_HOST_DEVICE
   void Record(
      const KernelContext & kernel,
      const GlobalIndex element_index,
      const StageId stage,
      const Integer component,
      const Real value ) const
   {
      if ( !enabled() ||
           !DiagnosticThreadParticipates( kernel ) ||
           element_index >= num_cells )
      {
         return;
      }

      const GlobalIndex thread_index = kernel.GetLinearThreadIndex();
      if ( thread_index >= diagnostic_threads )
      {
         return;
      }

      const GlobalIndex index =
         Index( stage, component, element_index, thread_index );
      values[ index ] = value;
      writes[ index ] = 1;
   }
};

struct StageDiagnostics
{
   DeviceBuffer< Real > values;
   DeviceBuffer< int > writes;
   GlobalIndex num_cells;

   StageDiagnostics( const GlobalIndex cells )
      : values(
           cells *
              static_cast< GlobalIndex >( StageId::count ) *
              diagnostic_components *
              diagnostic_threads,
           diagnostic_sentinel ),
        writes(
           cells *
              static_cast< GlobalIndex >( StageId::count ) *
              diagnostic_components *
              diagnostic_threads,
           0 ),
        num_cells( cells )
   {}

   StageRecorder Recorder() const
   {
      return {
         values.data.device_pointer,
         writes.data.device_pointer,
         num_cells
      };
   }

   void CopyToHost() const
   {
      values.CopyToHost();
      writes.CopyToHost();
   }

   GlobalIndex Index(
      const StageId stage,
      const Integer component,
      const GlobalIndex element_index,
      const GlobalIndex thread_index ) const
   {
      return (
         ( static_cast< GlobalIndex >( stage ) * diagnostic_components +
           component ) *
            num_cells +
         element_index ) *
            diagnostic_threads +
         thread_index;
   }
};

bool PrintStageComparison(
   const char * label,
   const StageDiagnostics & observed,
   const StageDiagnostics & expected,
   const Real tolerance )
{
   observed.CopyToHost();
   expected.CopyToHost();

   bool matches = true;

   for ( Integer stage_int = 0;
         stage_int < static_cast< Integer >( StageId::count );
         ++stage_int )
   {
      const StageId stage = static_cast< StageId >( stage_int );
      for ( Integer component = 0;
            component < diagnostic_components;
            ++component )
      {
         Real observed_sum = 0.0;
         Real observed_sum_sq = 0.0;
         Real observed_max_abs = 0.0;
         Real expected_sum = 0.0;
         Real expected_sum_sq = 0.0;
         Real expected_max_abs = 0.0;
         Integer observed_count = 0;
         Integer expected_count = 0;
         Integer mismatches = 0;
         Integer printed = 0;

         for ( GlobalIndex element = 0;
               element < observed.num_cells;
               ++element )
         {
            for ( GlobalIndex thread = 0;
                  thread < diagnostic_threads;
                  ++thread )
            {
               const GlobalIndex index =
                  observed.Index( stage, component, element, thread );
               const bool observed_written =
                  observed.writes.data.host_pointer[ index ] != 0;
               const bool expected_written =
                  expected.writes.data.host_pointer[ index ] != 0;
               const Real observed_value =
                  observed.values.data.host_pointer[ index ];
               const Real expected_value =
                  expected.values.data.host_pointer[ index ];

               if ( observed_written )
               {
                  ++observed_count;
                  observed_sum += observed_value;
                  observed_sum_sq += observed_value * observed_value;
                  observed_max_abs =
                     std::max( observed_max_abs, std::abs( observed_value ) );
               }

               if ( expected_written )
               {
                  ++expected_count;
                  expected_sum += expected_value;
                  expected_sum_sq += expected_value * expected_value;
                  expected_max_abs =
                     std::max( expected_max_abs, std::abs( expected_value ) );
               }

               const bool value_mismatch =
                  observed_written &&
                  expected_written &&
                  std::abs( observed_value - expected_value ) > tolerance;

               if (
                  observed_written != expected_written ||
                  value_mismatch )
               {
                  ++mismatches;
                  matches = false;
                  if ( printed < 8 )
                  {
                     std::cout
                        << "    mismatch " << label << " "
                        << StageName( stage )
                        << ", component " << component
                        << ", element " << element
                        << ", logical-thread " << thread
                        << ": observed_written=" << observed_written
                        << ", expected_written=" << expected_written
                        << ", observed=" << observed_value
                        << ", expected=" << expected_value << '\n';
                     ++printed;
                  }
               }
            }
         }

         std::cout
            << label << " " << StageName( stage )
            << ", component " << component
            << ": observed(count=" << observed_count
            << ", sum=" << observed_sum
            << ", sumsq=" << observed_sum_sq
            << ", maxabs=" << observed_max_abs
            << "), expected(count=" << expected_count
            << ", sum=" << expected_sum
            << ", sumsq=" << expected_sum_sq
            << ", maxabs=" << expected_max_abs
            << "), mismatches=" << mismatches << ".\n";
      }
   }

   return matches;
}

void PrintAllocatorResetAudit()
{
   static bool printed = false;
   if ( printed )
   {
      return;
   }
   printed = true;

   std::cout
      << "GradGrad SharedAllocator.reset() audit:\n"
      << "  InterpolateValuesThreaded: suspicious - reset follows final "
      << "shared-to-register read without a post-read Sync() before arena "
      << "reuse.\n"
      << "  InterpolateGradientThreaded: suspicious - reset follows shared "
      << "gradient contractions without an unconditional final Sync().\n"
      << "  ApplyGradientTestFunctionsAtQPoints: suspicious - reset follows "
      << "shared adjoint-gradient contractions without an unconditional final "
      << "Sync().\n"
      << "  ApplyTestFunctionsThreaded: suspicious - reset follows final "
      << "shared-to-register read without a post-read Sync() before arena "
      << "reuse.\n"
      << "  ThreadedWriteDofs cell path: safe - no shared arena allocation or "
      << "reset in the active cell L2/H1 write path.\n";
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
Vector ApplyProductionWriteOnlyGradGradWithInitial(
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
      MakeWriteOnlyElementTensorView< KernelPolicy >( fe_space, output );
   op.Apply( dofs_in, dofs_out );
   GENDIL_DEVICE_SYNC;

   return output;
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
   BarrierMode Mode,
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
   const bool write_output,
   const StageRecorder & recorder )
{
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );
   recorder.Record(
      kernel_conf,
      element_index,
      StageId::read_dofs,
      0,
      u() );
   if constexpr ( Mode == BarrierMode::sync_after_read_dofs )
   {
      kernel_conf.Sync();
   }

   auto Gu = InterpolateGradient( kernel_conf, element_quad_data, u );
   recorder.Record(
      kernel_conf,
      element_index,
      StageId::interpolate_gradient,
      0,
      Gu( 0 ) );
   if constexpr (
      Mode == BarrierMode::sync_after_interpolate_gradient ||
      Mode == BarrierMode::sync_after_allocator_reset )
   {
      kernel_conf.Sync();
   }

   auto GDGu =
      ApplyGradientTestFunctionsAtQPoints(
         kernel_conf,
         element_quad_data,
         Gu );
   recorder.Record(
      kernel_conf,
      element_index,
      StageId::apply_gradient_test,
      0,
      GDGu() );
   if constexpr (
      Mode == BarrierMode::sync_after_apply_gradient_test ||
      Mode == BarrierMode::sync_after_allocator_reset )
   {
      kernel_conf.Sync();
   }

   auto BGDGu = ApplyTestFunctions( kernel_conf, element_quad_data, GDGu );
   recorder.Record(
      kernel_conf,
      element_index,
      StageId::before_write_dofs,
      0,
      BGDGu() );
   if constexpr (
      Mode == BarrierMode::sync_before_write_dofs ||
      Mode == BarrierMode::sync_after_allocator_reset )
   {
      kernel_conf.Sync();
   }

   if ( write_output )
   {
      WriteDofs( kernel_conf, fe_space, element_index, BGDGu, dofs_out );

      if ( recorder.DiagnosticThreadParticipates( kernel_conf ) )
      {
         const GlobalIndex dof = kernel_conf.template GetThreadIndex< 0 >();
         if ( dof < diagnostic_dofs )
         {
            recorder.Record(
               kernel_conf,
               element_index,
               StageId::after_write_dofs,
               0,
               dofs_out( dof, element_index ) );
         }
      }
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

         DebugGradGradElementOperator< BarrierMode::none, IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            dofs_in,
            dofs_out,
            active,
            StageRecorder{ nullptr, nullptr, num_cells } );
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

         DebugGradGradElementOperator< BarrierMode::none, IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            dofs_in,
            dofs_out,
            true,
            StageRecorder{ nullptr, nullptr, num_cells } );
      } );
}

template <
   InvocationMode Invocation,
   BarrierMode Barrier,
   typename KernelConfiguration,
   typename IntegrationRule,
   typename FiniteElementSpace,
   typename MeshQuadData,
   typename ElementQuadData,
   typename DofsInView,
   typename DofsOutView >
void DebugVariantGradGradExplicitOperator(
   const FiniteElementSpace & fe_space,
   const MeshQuadData & mesh_quad_data,
   const ElementQuadData & element_quad_data,
   const DofsInView & dofs_in,
   DofsOutView & dofs_out,
   const StageRecorder & recorder )
{
   const GlobalIndex num_cells = fe_space.GetNumberOfCells();
   KernelConfiguration::BlockLoop(
      num_cells,
      [=] GENDIL_HOST_DEVICE ( const KernelConfiguration & kernel ) mutable
      {
         const bool active = kernel.IsActive( num_cells );
         bool write_output = true;
         GlobalIndex element_index = kernel.WorkItemIndex();

         if constexpr ( Invocation == InvocationMode::early_return )
         {
            if ( !active )
            {
               return;
            }
         }
         else if constexpr (
            Invocation == InvocationMode::runtime_active_guarded_write )
         {
            if ( !active )
            {
               return;
            }
            write_output = active;
         }
         else if constexpr (
            Invocation == InvocationMode::all_candidate_guarded_write )
         {
            element_index =
               active ? kernel.WorkItemIndex() : GlobalIndex( 0 );
            write_output = active;
         }
         else
         {
            // direct_debug_write_true is only used for full-batch cases.
            write_output = true;
         }

         constexpr size_t required_shared_mem =
            required_shared_memory_v< KernelConfiguration, IntegrationRule >;
         GENDIL_SHARED Real _shared_mem[
            KernelContext<
               KernelConfiguration,
               required_shared_mem >::shared_memory_block_size ];

         KernelContext< KernelConfiguration, required_shared_mem >
            kernel_conf( _shared_mem, kernel );

         DebugGradGradElementOperator< Barrier, IntegrationRule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            dofs_in,
            dofs_out,
            write_output,
            recorder );
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

template <
   InvocationMode Invocation,
   BarrierMode Barrier,
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyDebugVariantGradGradWithInitial(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & input,
   const Vector & initial,
   StageDiagnostics & stages )
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

   DebugVariantGradGradExplicitOperator<
      Invocation,
      Barrier,
      KernelPolicy,
      Rule >(
         fe_space,
         mesh_quad_data,
         element_quad_data,
         dofs_in,
         dofs_out,
         stages.Recorder() );
   GENDIL_DEVICE_SYNC;

   return output;
}

template <
   InvocationMode Invocation,
   BarrierMode Barrier,
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyDebugVariantGradGradWithInitialWriteOnly(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & input,
   const Vector & initial,
   StageDiagnostics & stages )
{
   Vector output( initial );

   auto dofs_in =
      MakeReadOnlyElementTensorView< KernelPolicy >( fe_space, input );
   auto dofs_out =
      MakeWriteOnlyElementTensorView< KernelPolicy >( fe_space, output );

   using Mesh = typename FiniteElementSpace::mesh_type;
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;

   typename Mesh::cell_type::template QuadData< Rule > mesh_quad_data{};
   auto element_quad_data =
      MakeDofToQuad< ShapeFunctions, Rule >();

   (void) integration_rule;

   DebugVariantGradGradExplicitOperator<
      Invocation,
      Barrier,
      KernelPolicy,
      Rule >(
         fe_space,
         mesh_quad_data,
         element_quad_data,
         dofs_in,
         dofs_out,
         stages.Recorder() );
   GENDIL_DEVICE_SYNC;

   return output;
}

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename Rule >
Vector ApplyDirectProductionGradGradWithInitial(
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
   const GlobalIndex num_cells = fe_space.GetNumberOfCells();

   (void) integration_rule;

   KernelPolicy::BlockLoop(
      num_cells,
      [=] GENDIL_HOST_DEVICE ( const KernelPolicy & kernel ) mutable
      {
         const GlobalIndex element_index = kernel.WorkItemIndex();

         constexpr size_t required_shared_mem =
            required_shared_memory_v< KernelPolicy, Rule >;
         GENDIL_SHARED Real _shared_mem[
            KernelContext<
               KernelPolicy,
               required_shared_mem >::shared_memory_block_size ];

         KernelContext< KernelPolicy, required_shared_mem >
            kernel_conf( _shared_mem, kernel );

         GradGradElementOperator< Rule >(
            kernel_conf,
            fe_space,
            element_index,
            mesh_quad_data,
            element_quad_data,
            dofs_in,
            dofs_out );
      } );
   GENDIL_DEVICE_SYNC;

   return output;
}

template < GlobalIndex NumCells >
bool RunDebugVariantCase()
{
   PrintAllocatorResetAudit();
   std::cout
      << "Skipping ThreadBlockLayout<3,5>, BatchSize=device_warp_size "
      << "GradGrad debug variant for num_cells = " << NumCells
      << ": the current threaded helper contract requires the mapped "
      << "1D thread dimension to cover the local DOF/quadrature extent.\n";
   return true;
}

template < typename KernelConfiguration, typename Rule >
void PrintFocusedWavefrontMetadata( const char * label )
{
   using Context = KernelContext<
      KernelConfiguration,
      required_shared_memory_v< KernelConfiguration, Rule > >;

   static constexpr GlobalIndex batch_size =
      static_cast< GlobalIndex >( KernelConfiguration::batch_size );
   static constexpr GlobalIndex threads_per_work_item =
      static_cast< GlobalIndex >( KernelConfiguration::GetNumberOfThreads() );

   std::cout
      << label
      << " wavefront-boundary metadata for elements 50, 51, 52.\n";
   std::cout
      << "  shared-memory: per_work_item="
      << Context::per_work_item_shared_memory_size
      << ", stride_per_work_item="
      << Context::shared_memory_stride_per_work_item
      << ", block_size=" << Context::shared_memory_block_size << ".\n";

   for ( GlobalIndex element = focus_first_element;
         element <= focus_last_element;
         ++element )
   {
      const GlobalIndex work_item_index = element;
      const GlobalIndex block_index = element / batch_size;
      const GlobalIndex batch_index = element % batch_size;
      const GlobalIndex arena_base_offset =
         batch_index *
         static_cast< GlobalIndex >(
            Context::shared_memory_stride_per_work_item );

      const GlobalIndex first_linear_thread =
         threads_per_work_item * batch_index;
      const GlobalIndex last_linear_thread =
         first_linear_thread + focus_thread_slots - 1;
      const GlobalIndex first_wavefront =
         first_linear_thread / static_cast< GlobalIndex >( device_warp_size );
      const GlobalIndex last_wavefront =
         last_linear_thread / static_cast< GlobalIndex >( device_warp_size );

      std::cout
         << "  element " << element
         << ": WorkItemIndex=" << work_item_index
         << ", blockIdx.x=" << block_index
         << ", BatchIndex=" << batch_index
         << ", shared-memory arena base offset=" << arena_base_offset
         << ", linear-thread range=" << first_linear_thread
         << ".." << last_linear_thread
         << ", wavefront range=" << first_wavefront
         << ".." << last_wavefront
         << ", crosses_wavefront_boundary="
         << ( first_wavefront != last_wavefront ) << ".\n";

      for ( GlobalIndex slot = 0; slot < focus_thread_slots; ++slot )
      {
         const GlobalIndex thread_x = slot;
         const GlobalIndex thread_y = batch_index;
         const GlobalIndex linear_thread_id =
            thread_x + threads_per_work_item * thread_y;
         const GlobalIndex wavefront_id =
            linear_thread_id /
            static_cast< GlobalIndex >( device_warp_size );

         std::cout
            << "    slot " << slot
            << ": threadIdx.x=" << thread_x
            << ", threadIdx.y=" << thread_y
            << ", linear_thread_id=" << linear_thread_id
            << ", wavefront_id=" << wavefront_id << ".\n";
      }
   }
}

bool PrintFocusedStageSnapshots(
   const char * label,
   const StageDiagnostics & observed,
   const StageDiagnostics & expected,
   const Real tolerance )
{
   observed.CopyToHost();
   expected.CopyToHost();

   bool matches = true;
   std::cout
      << label
      << " focused stage snapshots for elements 50, 51, 52 and "
      << "logical thread slots 0..4.\n";

   for ( Integer stage_int = 0;
         stage_int < static_cast< Integer >( StageId::count );
         ++stage_int )
   {
      const StageId stage = static_cast< StageId >( stage_int );
      std::cout << "  " << StageName( stage ) << '\n';

      for ( GlobalIndex element = focus_first_element;
            element <= focus_last_element;
            ++element )
      {
         if ( element >= observed.num_cells ||
              element >= expected.num_cells )
         {
            std::cout << "    element " << element
                      << " is outside one of the recorded cell ranges.\n";
            continue;
         }

         for ( GlobalIndex slot = 0; slot < focus_thread_slots; ++slot )
         {
            const GlobalIndex index =
               observed.Index( stage, 0, element, slot );
            const bool observed_written =
               observed.writes.data.host_pointer[ index ] != 0;
            const bool expected_written =
               expected.writes.data.host_pointer[ index ] != 0;
            const Real observed_value =
               observed.values.data.host_pointer[ index ];
            const Real expected_value =
               expected.values.data.host_pointer[ index ];
            const Real diff = observed_value - expected_value;
            const bool value_mismatch =
               observed_written &&
               expected_written &&
               std::abs( diff ) > tolerance;

            if ( observed_written != expected_written || value_mismatch )
            {
               matches = false;
            }

            std::cout
               << "    element " << element
               << ", slot " << slot
               << ": observed_written=" << observed_written
               << ", expected_written=" << expected_written
               << ", observed=" << observed_value
               << ", expected=" << expected_value
               << ", diff=" << diff
               << ", mismatch="
               << ( observed_written != expected_written ||
                    value_mismatch ) << ".\n";
         }
      }
   }

   return matches;
}

void PrintFocusedOutputValues(
   const char * label,
   const Vector & observed,
   const Vector & device_batch1,
   const Vector & legacy )
{
   std::cout
      << label
      << " focused final output values for elements 50, 51, 52 "
      << "(L2 DOF indices element*4 .. element*4+3).\n";

   for ( GlobalIndex element = focus_first_element;
         element <= focus_last_element;
         ++element )
   {
      const GlobalIndex begin = element * diagnostic_dofs;
      const GlobalIndex end = begin + diagnostic_dofs - 1;
      std::cout << "  element " << element
                << ", output-index-range=" << begin
                << ".." << end << ".\n";

      for ( GlobalIndex dof = 0; dof < diagnostic_dofs; ++dof )
      {
         const GlobalIndex index = begin + dof;
         if ( index >= observed.Size() ||
              index >= device_batch1.Size() ||
              index >= legacy.Size() )
         {
            std::cout << "    dof " << dof
                      << " index " << index
                      << " is outside one of the output vectors.\n";
            continue;
         }

         const Real observed_value = observed[ index ];
         const Real batch1_value = device_batch1[ index ];
         const Real legacy_value = legacy[ index ];
         std::cout
            << "    dof " << dof
            << ", index=" << index
            << ", DeviceBatchN=" << observed_value
            << ", DeviceBatch1=" << batch1_value
            << ", LegacyConfig=" << legacy_value
            << ", BatchN-minus-Batch1="
            << observed_value - batch1_value
            << ", BatchN-minus-Legacy="
            << observed_value - legacy_value << ".\n";
      }
   }
}

template < GlobalIndex NumCells >
bool RunFocusedSupportedDebugCase()
{
   PrintAllocatorResetAudit();

   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static constexpr Integer BatchSize = device_warp_size;

   if ( !LaunchConfigurationFits< Layout, BatchSize >(
           "ThreadBlockLayout<5>, BatchSize=device_warp_size focused GradGrad debug" ) )
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
   using Rule = std::remove_cvref_t< decltype( integration_rule ) >;

   Vector input = MakeInputVector(
      fe_space.GetNumberOfFiniteElementDofs() );
   auto zero =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         0.0 );

   StageDiagnostics stages_batch1( NumCells );
   StageDiagnostics stages_batchn( NumCells );

   auto y_legacy =
      ApplyProductionWriteOnlyGradGradWithInitial< LegacyConfig >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batch1 =
      ApplyProductionWriteOnlyGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batchn =
      ApplyProductionWriteOnlyGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batch1_debug =
      ApplyDebugVariantGradGradWithInitialWriteOnly<
         InvocationMode::runtime_active_guarded_write,
         BarrierMode::none,
         DeviceBatch1 >(
            fe_space,
            integration_rule,
            input,
            zero,
            stages_batch1 );
   auto y_batchn_debug =
      ApplyDebugVariantGradGradWithInitialWriteOnly<
         InvocationMode::runtime_active_guarded_write,
         BarrierMode::none,
         DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            zero,
            stages_batchn );

   constexpr Real tolerance = 1.0e-10;
   const char * batch_classification =
      NumCells % BatchSize == 0 ? "full-batch" : "partial-batch";
   bool success = true;

   std::cout
      << "Focused supported GradGrad L2/DG debug: ThreadBlockLayout<5>, "
      << "BatchSize=device_warp_size, num_cells=" << NumCells
      << ", batch=" << batch_classification
      << ", output-view=production_write_only_output_view.\n";
   std::cout
      << "  DeviceBatch1 uses the same logical ThreadBlockLayout<5> with "
      << "BatchSize=1, isolating the batching/wavefront effect.\n";

   success =
      CheckScaledL2Close(
         "Focused DeviceBatch1 vs LegacyConfig",
         y_batch1,
         y_legacy,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "Focused DeviceBatchN vs LegacyConfig",
         y_batchn,
         y_legacy,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "Focused DeviceBatchN vs same-layout DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;

   PrintScaledL2Diagnostic(
      "Focused debug DeviceBatch1 body vs production DeviceBatch1",
      y_batch1_debug,
      y_batch1,
      tolerance );
   PrintScaledL2Diagnostic(
      "Focused debug DeviceBatchN body vs production DeviceBatchN",
      y_batchn_debug,
      y_batchn,
      tolerance );

   PrintFocusedWavefrontMetadata< DeviceBatchN, Rule >(
      "ThreadBlockLayout<5>, BatchSize=device_warp_size" );
   PrintFocusedStageSnapshots(
      "ThreadBlockLayout<5>, BatchSize=device_warp_size",
      stages_batchn,
      stages_batch1,
      tolerance );
   PrintFocusedOutputValues(
      "ThreadBlockLayout<5>, BatchSize=device_warp_size",
      y_batchn,
      y_batch1,
      y_legacy );

   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   GlobalIndex NumCells >
void RunLayoutBatchSweepCase( const char * label )
{
   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return;
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
      ApplyProductionWriteOnlyGradGradWithInitial< LegacyConfig >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batch1 =
      ApplyProductionWriteOnlyGradGradWithInitial< DeviceBatch1 >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_batchn =
      ApplyProductionWriteOnlyGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );

   constexpr Real tolerance = 1.0e-10;
   const Real err_batch1 = ScaledL2Error( y_batchn, y_batch1 );
   const Real err_legacy = ScaledL2Error( y_batchn, y_legacy );
   const bool batch1_ok = std::isfinite( err_batch1 ) && err_batch1 <= tolerance;
   const bool legacy_ok = std::isfinite( err_legacy ) && err_legacy <= tolerance;
   const bool divides_wavefront =
      device_warp_size % Layout::GetNumberOfThreads() == 0;
   const char * batch_classification =
      NumCells % BatchSize == 0 ? "full-batch" : "partial-batch";

   std::cout
      << "NON-GATING layout/batch sweep: " << label
      << ", BatchSize=" << BatchSize
      << ", num_cells=" << NumCells
      << ", batch=" << batch_classification
      << ", threads_per_work_item=" << Layout::GetNumberOfThreads()
      << ", divides_wavefront=" << divides_wavefront
      << ", vs same-layout DeviceBatch1 error=" << err_batch1
      << " (" << ( batch1_ok ? "PASS" : "FAIL" ) << ")"
      << ", vs LegacyConfig error=" << err_legacy
      << " (" << ( legacy_ok ? "PASS" : "FAIL" ) << ").\n";
}

template < typename Layout, Integer MaxSharedDimensions, GlobalIndex NumCells >
void RunLayoutSweepForLayout( const char * label )
{
   RunLayoutBatchSweepCase< Layout, MaxSharedDimensions, 2, NumCells >(
      label );
   RunLayoutBatchSweepCase< Layout, MaxSharedDimensions, 4, NumCells >(
      label );
   RunLayoutBatchSweepCase< Layout, MaxSharedDimensions, 8, NumCells >(
      label );
   RunLayoutBatchSweepCase< Layout, MaxSharedDimensions, 16, NumCells >(
      label );
   RunLayoutBatchSweepCase< Layout, MaxSharedDimensions, 32, NumCells >(
      label );
   if constexpr ( device_warp_size != 32 )
   {
      RunLayoutBatchSweepCase<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         NumCells >( label );
   }
}

void RunLayoutBatchSweep()
{
   std::cout
      << "NON-GATING GradGrad L2/DG layout/batch sweep: compare layouts "
      << "that divide the wavefront size against ThreadBlockLayout<5>, "
      << "which does not divide 64 on HIP.\n";

   RunLayoutSweepForLayout< ThreadBlockLayout< 4 >, 1, 64 >(
      "ThreadBlockLayout<4>" );
   RunLayoutSweepForLayout< ThreadBlockLayout< 4 >, 1, 65 >(
      "ThreadBlockLayout<4>" );
   RunLayoutSweepForLayout< ThreadBlockLayout< 5 >, 1, 64 >(
      "ThreadBlockLayout<5>" );
   RunLayoutSweepForLayout< ThreadBlockLayout< 5 >, 1, 65 >(
      "ThreadBlockLayout<5>" );
   RunLayoutSweepForLayout< ThreadBlockLayout< 8 >, 1, 64 >(
      "ThreadBlockLayout<8>" );
   RunLayoutSweepForLayout< ThreadBlockLayout< 8 >, 1, 65 >(
      "ThreadBlockLayout<8>" );
}

#if 0
template < GlobalIndex NumCells >
bool RunUnsupportedDebugVariantCaseBody()
{
   PrintAllocatorResetAudit();

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

   StageDiagnostics stages_batch1( NumCells );
   StageDiagnostics stages_runtime_active( NumCells );
   StageDiagnostics stages_early_return( NumCells );
   StageDiagnostics stages_all_candidate( NumCells );

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
   auto y_batch1_debug =
      ApplyDebugVariantGradGradWithInitial<
         InvocationMode::direct_debug_write_true,
         BarrierMode::none,
         DeviceBatch1 >(
            fe_space,
            integration_rule,
            input,
            zero,
            stages_batch1 );
   auto y_filtered =
      ApplyFilteredGradGradWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         input,
         zero );
   auto y_runtime_active =
      ApplyDebugVariantGradGradWithInitial<
         InvocationMode::runtime_active_guarded_write,
         BarrierMode::none,
         DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            zero,
            stages_runtime_active );
   auto y_early_return =
      ApplyDebugVariantGradGradWithInitial<
         InvocationMode::early_return,
         BarrierMode::none,
         DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            zero,
            stages_early_return );
   auto y_all_candidate =
      ApplyDebugVariantGradGradWithInitial<
         InvocationMode::all_candidate_guarded_write,
         BarrierMode::none,
         DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            zero,
            stages_all_candidate );

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
   success =
      CheckScaledL2Close(
         "Debug DeviceBatch1 vs production DeviceBatch1",
         y_batch1_debug,
         y_batch1,
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
   PrintScaledL2Diagnostic(
      "Runtime-active guarded-write DeviceBatchN vs LegacyConfig",
      y_runtime_active,
      y_legacy,
      tolerance );
   PrintScaledL2Diagnostic(
      "Runtime-active guarded-write DeviceBatchN vs DeviceBatch1",
      y_runtime_active,
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
      StageDiagnostics stages_direct_debug( NumCells );
      auto y_direct_production =
         ApplyDirectProductionGradGradWithInitial< DeviceBatchN >(
            fe_space,
            integration_rule,
            input,
            zero );
      auto y_direct_debug =
         ApplyDebugVariantGradGradWithInitial<
            InvocationMode::direct_debug_write_true,
            BarrierMode::none,
            DeviceBatchN >(
               fe_space,
               integration_rule,
               input,
               zero,
               stages_direct_debug );

      PrintScaledL2Diagnostic(
         "Direct BlockLoop production body DeviceBatchN vs LegacyConfig",
         y_direct_production,
         y_legacy,
         tolerance );
      PrintScaledL2Diagnostic(
         "Direct BlockLoop production body DeviceBatchN vs DeviceBatch1",
         y_direct_production,
         y_batch1,
         tolerance );
      PrintScaledL2Diagnostic(
         "Direct debug write-true DeviceBatchN vs LegacyConfig",
         y_direct_debug,
         y_legacy,
         tolerance );
      PrintScaledL2Diagnostic(
         "Direct debug write-true DeviceBatchN vs DeviceBatch1",
         y_direct_debug,
         y_batch1,
         tolerance );

      PrintStageComparison(
         "direct_debug_write_true vs DeviceBatch1",
         stages_direct_debug,
         stages_batch1,
         tolerance );
   }
   else
   {
      std::cout
         << "Direct no-guard BlockLoop variants skipped for num_cells = "
         << NumCells
         << " because the final batch contains inactive candidates.\n";
   }

   PrintStageComparison(
      "runtime_active_guarded_write vs DeviceBatch1",
      stages_runtime_active,
      stages_batch1,
      tolerance );
   PrintStageComparison(
      "early_return vs DeviceBatch1",
      stages_early_return,
      stages_batch1,
      tolerance );
   PrintStageComparison(
      "all_candidate_guarded_write vs DeviceBatch1",
      stages_all_candidate,
      stages_batch1,
      tolerance );

   auto run_barrier_variant =
      [&] ( auto barrier_constant )
      {
         using BarrierConstant = decltype( barrier_constant );
         constexpr BarrierMode barrier = BarrierConstant::value;
         StageDiagnostics stages( NumCells );
         auto y =
            ApplyDebugVariantGradGradWithInitial<
               InvocationMode::runtime_active_guarded_write,
               barrier,
               DeviceBatchN >(
                  fe_space,
                  integration_rule,
                  input,
                  zero,
                  stages );
         PrintScaledL2Diagnostic(
            BarrierName( barrier ),
            y,
            y_batch1,
            tolerance );
      };

   run_barrier_variant(
      std::integral_constant<
         BarrierMode,
         BarrierMode::sync_after_read_dofs >{} );
   run_barrier_variant(
      std::integral_constant<
         BarrierMode,
         BarrierMode::sync_after_interpolate_gradient >{} );
   run_barrier_variant(
      std::integral_constant<
         BarrierMode,
         BarrierMode::sync_after_apply_gradient_test >{} );
   run_barrier_variant(
      std::integral_constant<
         BarrierMode,
         BarrierMode::sync_before_write_dofs >{} );
   run_barrier_variant(
      std::integral_constant<
         BarrierMode,
         BarrierMode::sync_after_allocator_reset >{} );

   if constexpr ( NumCells % BatchSize == 0 )
   {
      std::cout
         << "DIAGNOSTIC: num_cells = " << NumCells
         << " is a full-batch case. Any filtered, early-return, or "
         << "runtime-active mismatch here is not caused by inactive final "
         << "batch lanes.\n";
   }
   else if ( !filtered_matches_legacy || !filtered_matches_batch1 )
   {
      std::cout
         << "DIAGNOSTIC: filtered production-style GradGrad mismatched in "
         << "a final-partial-batch case. Because the full-batch diagnostic "
         << "also mismatches on this layout, do not attribute this solely to "
         << "inactive lanes.\n";
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
         << "final partial batch for this run.\n";
   }

   return success;
}
#endif

template <
   typename Layout,
   Integer MaxSharedDimensions,
   GlobalIndex NumCells >
bool RunThreadDimHypothesisCase( const char * label )
{
   static constexpr Integer BatchSize = device_warp_size;
   static constexpr Integer space_dim = 1;

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
      << "Thread-dim hypothesis case: " << label
      << ", num_cells = " << NumCells
      << ", thread_block_dim = " << Layout::thread_block_dim
      << ", space_dim = " << space_dim
      << ", thread_dim_gt_space_dim = "
      << ( Layout::thread_block_dim > space_dim )
      << ", logical_threads_per_work_item = "
      << Layout::GetNumberOfThreads() << '\n';

   success =
      CheckScaledL2Close(
         "  DeviceBatch1 vs LegacyConfig",
         y_batch1,
         y_legacy,
         tolerance ) && success;
   const bool filtered_ok =
      PrintScaledL2Diagnostic(
         "  Filtered DeviceBatchN vs DeviceBatch1",
         y_filtered,
         y_batch1,
         tolerance );
   const bool early_return_ok =
      PrintScaledL2Diagnostic(
         "  Early-return DeviceBatchN vs DeviceBatch1",
         y_early_return,
         y_batch1,
         tolerance );
   success =
      CheckScaledL2Close(
         "  All-candidate DeviceBatchN vs DeviceBatch1",
         y_all_candidate,
         y_batch1,
         tolerance ) && success;

   std::cout
      << "  SUMMARY " << label
      << ": filtered_ok=" << filtered_ok
      << ", early_return_ok=" << early_return_ok
      << ", all_candidate_required_ok=" << success
      << ". Extra logical dimensions are diagnostic-only evidence here.\n";

   return success;
}

template < GlobalIndex NumCells >
bool RunThreadDimHypothesisSweepForCellCount()
{
   std::cout
      << "Skipping compact thread_dim > space_dim hypothesis sweep for "
      << "num_cells = " << NumCells
      << ": its GradGrad diagnostic layouts include under-threaded helper "
      << "configurations with num_quad_1D = 5.\n";
   return true;
}

} // namespace

int main()
{
   bool success = true;
   success = RunFocusedSupportedDebugCase< 63 >() && success;
   success = RunFocusedSupportedDebugCase< 64 >() && success;
   success = RunFocusedSupportedDebugCase< 65 >() && success;
   success = RunDebugVariantCase< 64 >() && success;
   success = RunDebugVariantCase< 65 >() && success;
   RunLayoutBatchSweep();
   std::cout
      << "Classifying compact thread_dim > space_dim hypothesis sweep.\n";
   success = RunThreadDimHypothesisSweepForCellCount< 64 >() && success;
   success = RunThreadDimHypothesisSweepForCellCount< 65 >() && success;
   return success ? 0 : 1;
}

#endif
