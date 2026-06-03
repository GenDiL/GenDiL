// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <array>
#include <iostream>
#include <string>
#include <type_traits>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-face-speed-of-light-read-cell skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

#if defined( GENDIL_FACE_READ_CELL_DIRECT_GLOBAL_TEST ) && defined( GENDIL_FACE_READ_CELL_FULL_SHARED_TEST )
#error "Select only one batched FaceSpeedOfLight ReadCell test matrix."
#endif

#if !defined( GENDIL_FACE_READ_CELL_DIRECT_GLOBAL_TEST ) && !defined( GENDIL_FACE_READ_CELL_FULL_SHARED_TEST )
#error "Select the DirectGlobal or FullShared batched FaceSpeedOfLight ReadCell test matrix."
#endif

using namespace gendil;
using namespace gendil::test;

namespace
{
template < typename BaseKernelPolicy, typename FaceReadPolicy >
struct KernelPolicyWithFaceReadPolicy : public BaseKernelPolicy
{
   using face_read_dofs_policy = FaceReadPolicy;
};

template < typename BaseKernelPolicy, typename FaceReadPolicy >
using FaceReadKernelPolicy =
   std::conditional_t<
      std::is_same_v< FaceReadPolicy, DirectGlobalFaceReadDofsPolicy >,
      BaseKernelPolicy,
      KernelPolicyWithFaceReadPolicy< BaseKernelPolicy, FaceReadPolicy > >;

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule >
Vector ApplyFaceSpeedOfLightReadCell(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   const Vector & x )
{
   Vector y( fe_space.GetNumberOfFiniteElementDofs() );
   y = real_sentinel;

   auto op =
      MakeFaceSpeedOfLightOperator< KernelPolicy >(
         fe_space,
         integration_rule );
   op( x, y );
   GENDIL_DEVICE_SYNC;

   return y;
}

Vector MakeFaceInputVector( const Integer size )
{
   return Vector(
      size,
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.25 +
            0.0625 * static_cast< Real >( i ) +
            0.03125 * static_cast< Real >( ( i * 11 ) % 17 );
      } );
}

std::string Make1DCaseLabel(
   const char * label,
   const Integer num_cells,
   const char * cell_case_kind )
{
   return std::string( label ) +
      ", dim=1" +
      ", mesh_extents={" +
      std::to_string( num_cells ) +
      "}" +
      ", cell_case=" +
      cell_case_kind;
}

std::string Make2DCaseLabel(
   const char * label,
   const Integer num_cells_x,
   const Integer num_cells_y,
   const char * cell_case_kind )
{
   return std::string( label ) +
      ", dim=2" +
      ", mesh_extents={" +
      std::to_string( num_cells_x ) +
      "," +
      std::to_string( num_cells_y ) +
      "}" +
      ", cell_case=" +
      cell_case_kind;
}

std::string MakeZeroWorkItemCaseLabel( const char * label )
{
   return std::string( label ) +
      ", dim=none" +
      ", mesh_extents={}" +
      ", cell_case=zero-work-items";
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   typename FiniteElementSpace,
   typename Rule >
bool RunFaceReadCellComparison(
   const char * label,
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule )
{
   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return true;
   }

   using LegacyBase =
      ThreadFirstKernelConfiguration< Layout, MaxSharedDimensions >;
   using DeviceBatch1Base =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchNBase =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   using LegacyPolicy =
      FaceReadKernelPolicy< LegacyBase, FaceReadPolicy >;
   using DeviceBatch1Policy =
      FaceReadKernelPolicy< DeviceBatch1Base, FaceReadPolicy >;
   using DeviceBatchNPolicy =
      FaceReadKernelPolicy< DeviceBatchNBase, FaceReadPolicy >;

   std::cout << "Running " << label
             << ", num_cells=" << fe_space.GetNumberOfCells()
             << '\n'
             << std::flush;

   Vector x = MakeFaceInputVector( fe_space.GetNumberOfFiniteElementDofs() );

   const Vector y_legacy =
      ApplyFaceSpeedOfLightReadCell< LegacyPolicy >(
         fe_space,
         integration_rule,
         x );
   const Vector y_batch1 =
      ApplyFaceSpeedOfLightReadCell< DeviceBatch1Policy >(
         fe_space,
         integration_rule,
         x );
   const Vector y_batchn =
      ApplyFaceSpeedOfLightReadCell< DeviceBatchNPolicy >(
         fe_space,
         integration_rule,
         x );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   success =
      CheckNoValue( "LegacyConfig output", y_legacy, real_sentinel ) &&
      success;
   success =
      CheckNoValue( "DeviceBatch1 output", y_batch1, real_sentinel ) &&
      success;
   success =
      CheckNoValue( "DeviceBatchN output", y_batchn, real_sentinel ) &&
      success;
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

template <
   Integer BatchSize,
   typename FaceReadPolicy,
   typename FiniteElementSpace,
   typename Rule >
bool RunRegisterOnlyFaceReadCellComparison(
   const char * label,
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule )
{
   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 0;

   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return true;
   }

   using DeviceBatch1Base =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchNBase =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   using DeviceBatch1Policy =
      FaceReadKernelPolicy< DeviceBatch1Base, FaceReadPolicy >;
   using DeviceBatchNPolicy =
      FaceReadKernelPolicy< DeviceBatchNBase, FaceReadPolicy >;

   std::cout << "Running " << label
             << ", num_cells=" << fe_space.GetNumberOfCells()
             << '\n'
             << std::flush;

   Vector x = MakeFaceInputVector( fe_space.GetNumberOfFiniteElementDofs() );

   const Vector y_batch1 =
      ApplyFaceSpeedOfLightReadCell< DeviceBatch1Policy >(
         fe_space,
         integration_rule,
         x );
   const Vector y_batchn =
      ApplyFaceSpeedOfLightReadCell< DeviceBatchNPolicy >(
         fe_space,
         integration_rule,
         x );

   constexpr Real tolerance = 1.0e-10;
   bool success = true;
   success =
      CheckNoValue( "DeviceBatch1 output", y_batch1, real_sentinel ) &&
      success;
   success =
      CheckNoValue( "DeviceBatchN output", y_batchn, real_sentinel ) &&
      success;
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
   typename FaceReadPolicy >
bool CheckZeroWorkItemsForPolicies( const char * label )
{
   using LegacyBase =
      ThreadFirstKernelConfiguration< Layout, MaxSharedDimensions >;
   using DeviceBatch1Base =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchNBase =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   using LegacyPolicy =
      FaceReadKernelPolicy< LegacyBase, FaceReadPolicy >;
   using DeviceBatch1Policy =
      FaceReadKernelPolicy< DeviceBatch1Base, FaceReadPolicy >;
   using DeviceBatchNPolicy =
      FaceReadKernelPolicy< DeviceBatchNBase, FaceReadPolicy >;

   bool success = true;
   success =
      CheckZeroWorkItems< LegacyPolicy >( label, integer_sentinel ) &&
      success;
   success =
      CheckZeroWorkItems< DeviceBatch1Policy >( label, integer_sentinel ) &&
      success;
   success =
      CheckZeroWorkItems< DeviceBatchNPolicy >( label, integer_sentinel ) &&
      success;
   return success;
}

template < Integer BatchSize, typename FaceReadPolicy >
bool CheckRegisterOnlyZeroWorkItems( const char * label )
{
   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 0;
   using DeviceBatch1Base =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchNBase =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   using DeviceBatch1Policy =
      FaceReadKernelPolicy< DeviceBatch1Base, FaceReadPolicy >;
   using DeviceBatchNPolicy =
      FaceReadKernelPolicy< DeviceBatchNBase, FaceReadPolicy >;

   bool success = true;
   success =
      CheckZeroWorkItems< DeviceBatch1Policy >( label, integer_sentinel ) &&
      success;
   success =
      CheckZeroWorkItems< DeviceBatchNPolicy >( label, integer_sentinel ) &&
      success;
   return success;
}

template < Integer Order, Integer NumCells >
auto Make1DFaceSpace()
{
   static_assert( NumCells > 0 );
   const Real h = 1.0 / static_cast< Real >( NumCells );
   Cartesian1DMesh mesh( h, NumCells );

   FiniteElementOrders< Order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   return MakeFiniteElementSpace( mesh, finite_element );
}

template < Integer Order, Integer NumCellsX, Integer NumCellsY >
auto Make2DFaceSpace()
{
   static_assert( NumCellsX > 0 );
   static_assert( NumCellsY > 0 );
   const Real h =
      1.0 /
      static_cast< Real >(
         NumCellsX > NumCellsY ? NumCellsX : NumCellsY );
   Cartesian2DMesh mesh( h, NumCellsX, NumCellsY );

   FiniteElementOrders< Order, Order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   return MakeFiniteElementSpace( mesh, finite_element );
}

template < Integer Dim, Integer Order >
auto MakeFaceIntegrationRule()
{
   static constexpr Integer num_quad_1d = Order + 2;
   if constexpr ( Dim == 1 )
   {
      IntegrationRuleNumPoints< num_quad_1d > num_quads;
      return MakeIntegrationRule( num_quads );
   }
   else
   {
      static_assert( Dim == 2 );
      IntegrationRuleNumPoints< num_quad_1d, num_quad_1d > num_quads;
      return MakeIntegrationRule( num_quads );
   }
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order,
   Integer NumCells >
bool Run1DFaceReadCellCase(
   const char * label,
   const char * cell_case_kind )
{
   auto fe_space = Make1DFaceSpace< Order, NumCells >();
   auto integration_rule = MakeFaceIntegrationRule< 1, Order >();
   const std::string case_label =
      Make1DCaseLabel( label, NumCells, cell_case_kind );
   return RunFaceReadCellComparison<
      Layout,
      MaxSharedDimensions,
      BatchSize,
      FaceReadPolicy >( case_label.c_str(), fe_space, integration_rule );
}

template <
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order,
   Integer NumCells >
bool RunRegisterOnly1DFaceReadCellCase(
   const char * label,
   const char * cell_case_kind )
{
   auto fe_space = Make1DFaceSpace< Order, NumCells >();
   auto integration_rule = MakeFaceIntegrationRule< 1, Order >();
   const std::string case_label =
      Make1DCaseLabel( label, NumCells, cell_case_kind );
   return RunRegisterOnlyFaceReadCellComparison<
      BatchSize,
      FaceReadPolicy >( case_label.c_str(), fe_space, integration_rule );
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order,
   Integer NumCellsX,
   Integer NumCellsY >
bool Run2DFaceReadCellCase(
   const char * label,
   const char * cell_case_kind )
{
   auto fe_space = Make2DFaceSpace< Order, NumCellsX, NumCellsY >();
   auto integration_rule = MakeFaceIntegrationRule< 2, Order >();
   const std::string case_label =
      Make2DCaseLabel(
         label,
         NumCellsX,
         NumCellsY,
         cell_case_kind );
   return RunFaceReadCellComparison<
      Layout,
      MaxSharedDimensions,
      BatchSize,
      FaceReadPolicy >( case_label.c_str(), fe_space, integration_rule );
}

template <
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order,
   Integer NumCellsX,
   Integer NumCellsY >
bool RunRegisterOnly2DFaceReadCellCase(
   const char * label,
   const char * cell_case_kind )
{
   auto fe_space = Make2DFaceSpace< Order, NumCellsX, NumCellsY >();
   auto integration_rule = MakeFaceIntegrationRule< 2, Order >();
   const std::string case_label =
      Make2DCaseLabel(
         label,
         NumCellsX,
         NumCellsY,
         cell_case_kind );
   return RunRegisterOnlyFaceReadCellComparison<
      BatchSize,
      FaceReadPolicy >( case_label.c_str(), fe_space, integration_rule );
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order >
bool Run1DExactFaceReadCellCases( const char * label )
{
   const std::string zero_label = MakeZeroWorkItemCaseLabel( label );
   bool success = true;
   success =
      CheckZeroWorkItemsForPolicies<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy >( zero_label.c_str() ) && success;
   success =
      Run1DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         1 >( label, "single-cell" ) && success;
   if constexpr ( BatchSize > 1 )
   {
      success =
         Run1DFaceReadCellCase<
            Layout,
            MaxSharedDimensions,
            BatchSize,
            FaceReadPolicy,
            Order,
            BatchSize - 1 >( label, "final-partial-batch-minus-one" ) &&
         success;
      success =
         Run1DFaceReadCellCase<
            Layout,
            MaxSharedDimensions,
            BatchSize,
            FaceReadPolicy,
            Order,
            BatchSize >( label, "full-batch" ) && success;
   }
   success =
      Run1DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize + 1 >( label, "final-partial-batch-plus-one" ) &&
      success;
   success =
      Run1DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         10 >( label, "fixed-ten-cells" ) && success;
   return success;
}

template < Integer BatchSize, typename FaceReadPolicy, Integer Order >
bool RunRegisterOnly1DExactFaceReadCellCases( const char * label )
{
   const std::string zero_label = MakeZeroWorkItemCaseLabel( label );
   bool success = true;
   success =
      CheckRegisterOnlyZeroWorkItems< BatchSize, FaceReadPolicy >(
         zero_label.c_str() ) && success;
   success =
      RunRegisterOnly1DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         1 >( label, "single-cell" ) && success;
   if constexpr ( BatchSize > 1 )
   {
      success =
         RunRegisterOnly1DFaceReadCellCase<
            BatchSize,
            FaceReadPolicy,
            Order,
            BatchSize - 1 >( label, "final-partial-batch-minus-one" ) &&
         success;
      success =
         RunRegisterOnly1DFaceReadCellCase<
            BatchSize,
            FaceReadPolicy,
            Order,
            BatchSize >( label, "full-batch" ) && success;
   }
   success =
      RunRegisterOnly1DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize + 1 >( label, "final-partial-batch-plus-one" ) &&
      success;
   success =
      RunRegisterOnly1DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         10 >( label, "fixed-ten-cells" ) && success;
   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order >
bool Run2DSkinnyFaceReadCellCases( const char * label )
{
   bool success = true;
   success =
      Run2DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         1,
         1 >( label, "skinny-single-cell" ) && success;
   if constexpr ( BatchSize > 1 )
   {
      success =
         Run2DFaceReadCellCase<
            Layout,
            MaxSharedDimensions,
            BatchSize,
            FaceReadPolicy,
            Order,
            BatchSize - 1,
            1 >( label, "skinny-final-partial-batch-minus-one" ) &&
         success;
      success =
         Run2DFaceReadCellCase<
            Layout,
            MaxSharedDimensions,
            BatchSize,
            FaceReadPolicy,
            Order,
            BatchSize,
            1 >( label, "skinny-full-batch" ) && success;
   }
   success =
      Run2DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize + 1,
         1 >( label, "skinny-final-partial-batch-plus-one" ) && success;
   success =
      Run2DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         10,
         1 >( label, "skinny-fixed-ten-cells" ) && success;
   return success;
}

template < Integer BatchSize, typename FaceReadPolicy, Integer Order >
bool RunRegisterOnly2DSkinnyFaceReadCellCases( const char * label )
{
   bool success = true;
   success =
      RunRegisterOnly2DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         1,
         1 >( label, "skinny-single-cell" ) && success;
   if constexpr ( BatchSize > 1 )
   {
      success =
         RunRegisterOnly2DFaceReadCellCase<
            BatchSize,
            FaceReadPolicy,
            Order,
            BatchSize - 1,
            1 >( label, "skinny-final-partial-batch-minus-one" ) &&
         success;
      success =
         RunRegisterOnly2DFaceReadCellCase<
            BatchSize,
            FaceReadPolicy,
            Order,
            BatchSize,
            1 >( label, "skinny-full-batch" ) && success;
   }
   success =
      RunRegisterOnly2DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize + 1,
         1 >( label, "skinny-final-partial-batch-plus-one" ) && success;
   success =
      RunRegisterOnly2DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         10,
         1 >( label, "skinny-fixed-ten-cells" ) && success;
   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order >
bool Run2DNondegenerateFaceReadCellCase( const char * label )
{
   return Run2DFaceReadCellCase<
      Layout,
      MaxSharedDimensions,
      BatchSize,
      FaceReadPolicy,
      Order,
      3,
      4 >( label, "nondegenerate-2d" );
}

template < Integer BatchSize, typename FaceReadPolicy, Integer Order >
bool RunRegisterOnly2DNondegenerateFaceReadCellCase( const char * label )
{
   return RunRegisterOnly2DFaceReadCellCase<
      BatchSize,
      FaceReadPolicy,
      Order,
      3,
      4 >( label, "nondegenerate-2d" );
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order >
bool RunThreadedLayoutCases( const char * label )
{
   bool success = true;
   success =
      Run1DExactFaceReadCellCases<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   success =
      Run2DSkinnyFaceReadCellCases<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   success =
      Run2DNondegenerateFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order >
bool RunThreaded2DLayoutCases( const char * label )
{
   bool success = true;
   success =
      Run2DSkinnyFaceReadCellCases<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   success =
      Run2DNondegenerateFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   return success;
}

template < Integer BatchSize, typename FaceReadPolicy, Integer Order >
bool RunRegisterOnlyLayoutCases( const char * label )
{
   bool success = true;
   success =
      RunRegisterOnly1DExactFaceReadCellCases<
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   success =
      RunRegisterOnly2DSkinnyFaceReadCellCases<
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   success =
      RunRegisterOnly2DNondegenerateFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   return success;
}

template < Integer BatchSize, typename FaceReadPolicy, Integer Order >
bool RunRegisterOnlyRepresentativeLayoutCases( const char * label )
{
   static_assert( BatchSize > 1 );
   bool success = true;
   success =
      RunRegisterOnly1DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize - 1 >(
            label,
            "final-partial-batch-minus-one" ) && success;
   success =
      RunRegisterOnly1DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize >( label, "full-batch" ) && success;
   success =
      RunRegisterOnly1DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize + 1 >(
            label,
            "final-partial-batch-plus-one" ) && success;

   success =
      RunRegisterOnly2DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize - 1,
         1 >(
            label,
            "skinny-final-partial-batch-minus-one" ) && success;
   success =
      RunRegisterOnly2DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize,
         1 >( label, "skinny-full-batch" ) && success;
   success =
      RunRegisterOnly2DFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize + 1,
         1 >(
            label,
            "skinny-final-partial-batch-plus-one" ) && success;
   success =
      RunRegisterOnly2DNondegenerateFaceReadCellCase<
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   typename FaceReadPolicy,
   Integer Order >
bool RunThreadedRepresentativeLayoutCases( const char * label )
{
   static_assert( BatchSize > 1 );
   bool success = true;
   success =
      Run1DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize - 1 >(
            label,
            "final-partial-batch-minus-one" ) && success;
   success =
      Run1DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize >( label, "full-batch" ) && success;
   success =
      Run1DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize + 1 >(
            label,
            "final-partial-batch-plus-one" ) && success;

   success =
      Run2DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize - 1,
         1 >(
            label,
            "skinny-final-partial-batch-minus-one" ) && success;
   success =
      Run2DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize,
         1 >( label, "skinny-full-batch" ) && success;
   success =
      Run2DFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order,
         BatchSize + 1,
         1 >(
            label,
            "skinny-final-partial-batch-plus-one" ) && success;
   success =
      Run2DNondegenerateFaceReadCellCase<
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FaceReadPolicy,
         Order >( label ) && success;
   return success;
}

template < Integer Order, typename FaceReadPolicy >
bool TestBroadReadPolicy( const char * policy_label )
{
   static constexpr Integer num_dofs_1d = Order + 1;
   static constexpr Integer num_quad_1d = Order + 2;
   bool success = true;

   auto make_label =
      [policy_label](
         const char * layout_label,
         const Integer batch_size,
         const char * oracle_label )
      {
         return std::string( "policy=" ) +
            policy_label +
            ", order=" +
            std::to_string( Order ) +
            ", " +
            layout_label +
            ", BatchSize=" +
            std::to_string( batch_size ) +
            ", oracle=" +
            oracle_label;
      };

   auto label_register_batch1 =
      make_label( "layout=ThreadBlockLayout<>", 1, "DeviceBatch1" );
   success =
      RunRegisterOnlyLayoutCases< 1, FaceReadPolicy, Order >(
         label_register_batch1.c_str() ) && success;
   auto label_register_batch2 =
      make_label( "layout=ThreadBlockLayout<>", 2, "DeviceBatch1" );
   success =
      RunRegisterOnlyLayoutCases< 2, FaceReadPolicy, Order >(
         label_register_batch2.c_str() ) && success;
   auto label_register_batch4 =
      make_label( "layout=ThreadBlockLayout<>", 4, "DeviceBatch1" );
   success =
      RunRegisterOnlyLayoutCases< 4, FaceReadPolicy, Order >(
         label_register_batch4.c_str() ) && success;
   auto label_register_batch_warp =
      make_label(
         "layout=ThreadBlockLayout<>",
         device_warp_size,
         "DeviceBatch1" );
   success =
      RunRegisterOnlyLayoutCases<
         device_warp_size,
         FaceReadPolicy,
         Order >(
         label_register_batch_warp.c_str() ) && success;

   using OneDofThreadedDim = ThreadBlockLayout< num_dofs_1d >;
   static constexpr Integer OneThreadedMaxSharedDimensions = 1;

   auto label_one_dof_threaded_batch1 =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d>",
         1,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedLayoutCases<
         OneDofThreadedDim,
         OneThreadedMaxSharedDimensions,
         1,
         FaceReadPolicy,
         Order >( label_one_dof_threaded_batch1.c_str() ) && success;
   auto label_one_dof_threaded_batch2 =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d>",
         2,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedLayoutCases<
         OneDofThreadedDim,
         OneThreadedMaxSharedDimensions,
         2,
         FaceReadPolicy,
         Order >( label_one_dof_threaded_batch2.c_str() ) && success;
   auto label_one_dof_threaded_batch4 =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d>",
         4,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedLayoutCases<
         OneDofThreadedDim,
         OneThreadedMaxSharedDimensions,
         4,
         FaceReadPolicy,
         Order >( label_one_dof_threaded_batch4.c_str() ) && success;
   auto label_one_dof_threaded_batch_warp =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d>",
         device_warp_size,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedLayoutCases<
         OneDofThreadedDim,
         OneThreadedMaxSharedDimensions,
         device_warp_size,
         FaceReadPolicy,
         Order >( label_one_dof_threaded_batch_warp.c_str() ) &&
      success;

   using OneQuadThreadedDim = ThreadBlockLayout< num_quad_1d >;

   auto label_one_quad_threaded_batch1 =
      make_label(
         "layout=ThreadBlockLayout<num_quad_1d>",
         1,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedLayoutCases<
         OneQuadThreadedDim,
         OneThreadedMaxSharedDimensions,
         1,
         FaceReadPolicy,
         Order >( label_one_quad_threaded_batch1.c_str() ) && success;
   auto label_one_quad_threaded_batch2 =
      make_label(
         "layout=ThreadBlockLayout<num_quad_1d>",
         2,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedLayoutCases<
         OneQuadThreadedDim,
         OneThreadedMaxSharedDimensions,
         2,
         FaceReadPolicy,
         Order >( label_one_quad_threaded_batch2.c_str() ) && success;
   auto label_one_quad_threaded_batch4 =
      make_label(
         "layout=ThreadBlockLayout<num_quad_1d>",
         4,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedLayoutCases<
         OneQuadThreadedDim,
         OneThreadedMaxSharedDimensions,
         4,
         FaceReadPolicy,
         Order >( label_one_quad_threaded_batch4.c_str() ) && success;
   auto label_one_quad_threaded_batch_warp =
      make_label(
         "layout=ThreadBlockLayout<num_quad_1d>",
         device_warp_size,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedLayoutCases<
         OneQuadThreadedDim,
         OneThreadedMaxSharedDimensions,
         device_warp_size,
         FaceReadPolicy,
         Order >( label_one_quad_threaded_batch_warp.c_str() ) &&
      success;

   using TwoDofThreadedDims =
      ThreadBlockLayout< num_dofs_1d, num_dofs_1d >;
   static constexpr Integer TwoThreadedMaxSharedDimensions = 2;

   auto label_two_dof_threaded_batch1 =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d,num_dofs_1d>",
         1,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreaded2DLayoutCases<
         TwoDofThreadedDims,
         TwoThreadedMaxSharedDimensions,
         1,
         FaceReadPolicy,
         Order >( label_two_dof_threaded_batch1.c_str() ) && success;
   auto label_two_dof_threaded_batch2 =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d,num_dofs_1d>",
         2,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreaded2DLayoutCases<
         TwoDofThreadedDims,
         TwoThreadedMaxSharedDimensions,
         2,
         FaceReadPolicy,
         Order >( label_two_dof_threaded_batch2.c_str() ) && success;
   auto label_two_dof_threaded_batch4 =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d,num_dofs_1d>",
         4,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreaded2DLayoutCases<
         TwoDofThreadedDims,
         TwoThreadedMaxSharedDimensions,
         4,
         FaceReadPolicy,
         Order >( label_two_dof_threaded_batch4.c_str() ) && success;
   auto label_two_dof_threaded_batch_warp =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d,num_dofs_1d>",
         device_warp_size,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreaded2DLayoutCases<
         TwoDofThreadedDims,
         TwoThreadedMaxSharedDimensions,
         device_warp_size,
         FaceReadPolicy,
         Order >( label_two_dof_threaded_batch_warp.c_str() ) &&
      success;

   using TwoQuadThreadedDims =
      ThreadBlockLayout< num_quad_1d, num_quad_1d >;

   auto label_two_quad_threaded_batch1 =
      make_label(
         "layout=ThreadBlockLayout<num_quad_1d,num_quad_1d>",
         1,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreaded2DLayoutCases<
         TwoQuadThreadedDims,
         TwoThreadedMaxSharedDimensions,
         1,
         FaceReadPolicy,
         Order >( label_two_quad_threaded_batch1.c_str() ) && success;
   auto label_two_quad_threaded_batch2 =
      make_label(
         "layout=ThreadBlockLayout<num_quad_1d,num_quad_1d>",
         2,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreaded2DLayoutCases<
         TwoQuadThreadedDims,
         TwoThreadedMaxSharedDimensions,
         2,
         FaceReadPolicy,
         Order >( label_two_quad_threaded_batch2.c_str() ) && success;
   auto label_two_quad_threaded_batch4 =
      make_label(
         "layout=ThreadBlockLayout<num_quad_1d,num_quad_1d>",
         4,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreaded2DLayoutCases<
         TwoQuadThreadedDims,
         TwoThreadedMaxSharedDimensions,
         4,
         FaceReadPolicy,
         Order >( label_two_quad_threaded_batch4.c_str() ) && success;
   auto label_two_quad_threaded_batch_warp =
      make_label(
         "layout=ThreadBlockLayout<num_quad_1d,num_quad_1d>",
         device_warp_size,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreaded2DLayoutCases<
         TwoQuadThreadedDims,
         TwoThreadedMaxSharedDimensions,
         device_warp_size,
         FaceReadPolicy,
         Order >( label_two_quad_threaded_batch_warp.c_str() ) &&
      success;

   return success;
}

template < Integer Order >
bool TestFullSharedRepresentativePolicy()
{
   static constexpr Integer num_dofs_1d = Order + 1;
   const char * policy_label = "FullSharedOverride";
   bool success = true;

   auto make_label =
      [policy_label](
         const char * layout_label,
         const Integer batch_size,
         const char * oracle_label )
      {
         return std::string( "policy=" ) +
            policy_label +
            ", order=" +
            std::to_string( Order ) +
            ", " +
            layout_label +
            ", BatchSize=" +
            std::to_string( batch_size ) +
            ", oracle=" +
            oracle_label;
      };

   auto register_batch2_label =
      make_label( "layout=ThreadBlockLayout<>", 2, "DeviceBatch1" );
   success =
      RunRegisterOnlyRepresentativeLayoutCases<
         2,
         FullSharedFaceReadDofsPolicy,
         Order >( register_batch2_label.c_str() ) && success;
   auto register_batch_warp_label =
      make_label(
         "layout=ThreadBlockLayout<>",
         device_warp_size,
         "DeviceBatch1" );
   success =
      RunRegisterOnlyRepresentativeLayoutCases<
         device_warp_size,
         FullSharedFaceReadDofsPolicy,
         Order >( register_batch_warp_label.c_str() ) && success;

   using OneThreadedDim = ThreadBlockLayout< num_dofs_1d >;
   static constexpr Integer OneThreadedMaxSharedDimensions = 1;

   auto threaded_batch2_label =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d>",
         2,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedRepresentativeLayoutCases<
         OneThreadedDim,
         OneThreadedMaxSharedDimensions,
         2,
         FullSharedFaceReadDofsPolicy,
         Order >( threaded_batch2_label.c_str() ) && success;
   auto threaded_batch_warp_label =
      make_label(
         "layout=ThreadBlockLayout<num_dofs_1d>",
         device_warp_size,
         "LegacyConfig+DeviceBatch1" );
   success =
      RunThreadedRepresentativeLayoutCases<
         OneThreadedDim,
         OneThreadedMaxSharedDimensions,
         device_warp_size,
         FullSharedFaceReadDofsPolicy,
         Order >( threaded_batch_warp_label.c_str() ) && success;

   return success;
}

} // namespace

int main()
{
   static constexpr Integer order = 3;
   bool success = true;

#if defined( GENDIL_FACE_READ_CELL_DIRECT_GLOBAL_TEST )
   success =
      TestBroadReadPolicy< order, DirectGlobalFaceReadDofsPolicy >(
         "DirectGlobalDefault" ) && success;
#elif defined( GENDIL_FACE_READ_CELL_FULL_SHARED_TEST )
   success =
      TestFullSharedRepresentativePolicy< order >() && success;
#endif

   return success ? 0 : 1;
}

#endif
