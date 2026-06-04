// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <array>
#include <iostream>
#include <utility>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-linear-form-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{
constexpr Real output_sentinel = -123456.75;

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule, typename Source >
Vector MakeLinearFormVector(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   Source & source )
{
   Vector rhs =
      MakeLinearForm< KernelPolicy >(
         fe_space,
         integration_rule,
         source );
   GENDIL_DEVICE_SYNC;
   return rhs;
}

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule, typename Source >
Vector ApplyLinearFormWithInitial(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   Source & source,
   const Vector & initial )
{
   using Mesh = typename FiniteElementSpace::mesh_type;
   using MeshQuadData =
      typename Mesh::cell_type::template QuadData< Rule >;
   using ElementQuadData =
      decltype(
         MakeDofToQuad<
            typename FiniteElementSpace::finite_element_type::shape_functions,
            Rule >() );

   Vector out( initial );
   MeshQuadData mesh_quad_data;
   ElementQuadData element_quad_data;
   auto dofs_out =
      MakeReadWriteElementTensorView< KernelPolicy >( fe_space, out );

   LinearFormOperator< KernelPolicy, Rule >(
      fe_space,
      mesh_quad_data,
      element_quad_data,
      source,
      dofs_out );
   GENDIL_DEVICE_SYNC;

   return out;
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

   auto source = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 0.75 + 0.5 * x + 0.125 * x * x;
   };

   auto rhs_batch1 =
      MakeLinearFormVector< DeviceBatch1 >(
         fe_space,
         integration_rule,
         source );
   auto rhs_batchn =
      MakeLinearFormVector< DeviceBatchN >(
         fe_space,
         integration_rule,
         source );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   std::cout << label << " L2, num_cells = " << num_cells << '\n';

   if constexpr ( CompareLegacy )
   {
      auto rhs_legacy =
         MakeLinearFormVector< LegacyConfig >(
            fe_space,
            integration_rule,
            source );
      success =
         CheckScaledL2Close(
            "DeviceBatchN vs LegacyConfig",
            rhs_batchn,
            rhs_legacy,
            tolerance ) && success;
      success =
         CheckScaledL2Close(
            "DeviceBatch1 vs LegacyConfig",
            rhs_batch1,
            rhs_legacy,
            tolerance ) && success;
   }

   success =
      CheckScaledL2Close(
         "DeviceBatchN vs DeviceBatch1",
         rhs_batchn,
         rhs_batch1,
         tolerance ) && success;

   auto zero = MakeConstantVector( fe_space.GetNumberOfFiniteElementDofs(), 0.0 );
   auto sentinel_initial =
      MakeConstantVector(
         fe_space.GetNumberOfFiniteElementDofs(),
         output_sentinel );
   auto direct_zero =
      ApplyLinearFormWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         source,
         zero );
   auto direct_sentinel =
      ApplyLinearFormWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         source,
         sentinel_initial );

   success =
      CheckScaledL2Close(
         "DeviceBatchN direct zero-baseline vs MakeLinearForm",
         direct_zero,
         rhs_batchn,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "DeviceBatchN L2 overwrite from sentinel",
         direct_sentinel,
         direct_zero,
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

   auto source = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 1.125 + 0.375 * x + 0.625 * x * x;
   };

   auto rhs_batch1 =
      MakeLinearFormVector< DeviceBatch1 >(
         fe_space,
         integration_rule,
         source );
   auto rhs_batchn =
      MakeLinearFormVector< DeviceBatchN >(
         fe_space,
         integration_rule,
         source );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   std::cout << label << " H1, num_cells = " << num_cells << '\n';

   if constexpr ( CompareLegacy )
   {
      auto rhs_legacy =
         MakeLinearFormVector< LegacyConfig >(
            fe_space,
            integration_rule,
            source );
      success =
         CheckScaledL2Close(
            "DeviceBatchN vs LegacyConfig",
            rhs_batchn,
            rhs_legacy,
            tolerance ) && success;
      success =
         CheckScaledL2Close(
            "DeviceBatch1 vs LegacyConfig",
            rhs_batch1,
            rhs_legacy,
            tolerance ) && success;
   }

   success =
      CheckScaledL2Close(
         "DeviceBatchN vs DeviceBatch1",
         rhs_batchn,
         rhs_batch1,
         tolerance ) && success;

   auto zero = MakeConstantVector( fe_space.GetNumberOfFiniteElementDofs(), 0.0 );
   auto baseline = MakeBaselineVector( fe_space.GetNumberOfFiniteElementDofs() );
   auto direct_zero =
      ApplyLinearFormWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         source,
         zero );
   auto direct_baseline =
      ApplyLinearFormWithInitial< DeviceBatchN >(
         fe_space,
         integration_rule,
         source,
         baseline );
   auto expected_accumulated = AddVectors( baseline, direct_zero );

   success =
      CheckScaledL2Close(
         "DeviceBatchN direct zero-baseline vs MakeLinearForm",
         direct_zero,
         rhs_batchn,
         tolerance ) && success;
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
bool RunLinearFormBatchCases( const char * label )
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

bool TestThreadedLinearForm()
{
   using Layout = ThreadBlockLayout< 5 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static_assert( Layout::GetNumberOfThreads() == 5 );

   bool success = true;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 1, true >(
         "ThreadBlockLayout<5>, BatchSize=1" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 2, true >(
         "ThreadBlockLayout<5>, BatchSize=2" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 4, true >(
         "ThreadBlockLayout<5>, BatchSize=4" ) && success;
   success =
      RunLinearFormBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         true >(
            "ThreadBlockLayout<5>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestRegisterOnlyLinearForm()
{
   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 0;

   bool success = true;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 1, false >(
         "ThreadBlockLayout<>, BatchSize=1" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 2, false >(
         "ThreadBlockLayout<>, BatchSize=2" ) && success;
   success =
      RunLinearFormBatchCases< Layout, MaxSharedDimensions, 4, false >(
         "ThreadBlockLayout<>, BatchSize=4" ) && success;
   success =
      RunLinearFormBatchCases<
         Layout,
         MaxSharedDimensions,
         device_warp_size,
         false >(
            "ThreadBlockLayout<>, BatchSize=device_warp_size" ) &&
      success;
   return success;
}

bool TestIrregularLinearFormDiagnostic()
{
   std::cout
      << "Skipping ThreadBlockLayout<3,5> LinearForm diagnostic: the current "
      << "threaded helper contract requires the mapped 1D thread dimension "
      << "to cover the local DOF/quadrature extent.\n";
   return true;
}

} // namespace

int main()
{
   bool success = true;
   success = TestThreadedLinearForm() && success;
   success = TestRegisterOnlyLinearForm() && success;
   success = TestIrregularLinearFormDiagnostic() && success;

   return success ? 0 : 1;
}

#endif
