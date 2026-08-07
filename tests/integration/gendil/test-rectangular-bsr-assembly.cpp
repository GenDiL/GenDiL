// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-12;

#if defined(GENDIL_USE_DEVICE)
using RectangularKernelPolicy =
   ThreadFirstKernelConfiguration< ThreadBlockLayout< 4, 4 >, 2 >;
#else
using RectangularKernelPolicy = SerialKernelConfiguration;
#endif

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool Near( const Real lhs, const Real rhs )
{
   return std::abs( lhs - rhs ) < tolerance;
}

void FillDeterministicInput( Vector & x )
{
   Real * data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      data[i] =
         0.35 +
         0.11 * static_cast< Real >( i ) +
         0.017 * static_cast< Real >( i * i );
   }
}

template < typename Operator >
Vector ApplyOperator(
   const Operator & op,
   const Vector & x,
   const GlobalIndex output_size )
{
   Vector result( output_size );
   result = 0.0;
   op( x, result );
   return result;
}

template < typename Matrix >
bool CheckAction(
   const Matrix & matrix,
   const Vector & x,
   const Vector & expected,
   const char * message )
{
   const auto result = ApplyOperator( matrix, x, expected.Size() );
   const Real * result_data = result.ReadHostData();
   const Real * expected_data = expected.ReadHostData();

   bool success = true;
   for ( GlobalIndex i = 0; i < expected.Size(); ++i )
   {
      success =
         Check( Near( result_data[i], expected_data[i] ), message ) &&
         success;
   }
   return success;
}

template < typename Matrix >
bool CheckSGBSRDirectPipeline(
   const Matrix & matrix,
   const Vector & x,
   const Vector & expected,
   const char * message )
{
   Vector x_bsr( matrix.TrialBsrSize() );
   Vector y_bsr( matrix.TestBsrSize() );
   Vector y_fe( matrix.NumRows() );
   matrix.trial_gather(
      matrix.bsr_matrix.backend,
      x,
      x_bsr );
   matrix.bsr_matrix( x_bsr, y_bsr );
   matrix.test_scatter(
      matrix.bsr_matrix.backend,
      y_bsr,
      y_fe );

   const Real * actual_data = y_fe.ReadHostData();
   const Real * expected_data = expected.ReadHostData();
   bool success = true;
   for ( GlobalIndex i = 0; i < expected.Size(); ++i )
   {
      success = Check(
         Near( actual_data[i], expected_data[i] ),
         message ) && success;
   }
   return success;
}

const Real * WorkspacePointer( const Vector & vector )
{
#if defined(GENDIL_USE_DEVICE)
   return vector.ReadDeviceData();
#else
   return vector.ReadHostData();
#endif
}

struct OrientedTwoQuadFixture
{
   // Cell 0 is axis-aligned. Cell 1 is rotated by 180 degrees, so both cells
   // are positively oriented but their shared-face tangential coordinates run
   // in opposite directions.
   std::array< Real, 12 > node_data{
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 0.0,
      2.0, 1.0 };
   std::array< int, 8 > restriction_data{
      0, 1, 2, 3,
      5, 3, 4, 1 };
   HostDevicePointer< const int > restriction_pointer{};
   UnstructuredConformingConnectivity< HyperCube< 2 > > connectivity{ 2 };

   OrientedTwoQuadFixture()
   {
      restriction_pointer.host_pointer = restriction_data.data();
      for ( GlobalIndex element = 0; element < 2; ++element )
      {
         for ( Integer face = 0; face < HyperCube< 2 >::num_faces; ++face )
         {
            connectivity[element].faces[face] =
               { 0,
                 {},
                 MakeReferencePermutation< 2 >(),
                 {},
                 {},
                 true };
         }
      }

      const Permutation< 2 > rotated_neighbor{ { -1, -2 } };
      connectivity[0].faces[2] =
         { 1, {}, rotated_neighbor, {}, {}, false };
      connectivity[1].faces[2] =
         { 0, {}, rotated_neighbor, {}, {}, false };
   }

   auto MakeMesh() const
   {
      return QuadMesh< 1 >{
         MakeFIFOView( node_data.data(), GlobalIndex( 2 ), GlobalIndex( 6 ) ),
         MakeFIFOView(
            restriction_pointer,
            GlobalIndex( 2 ),
            GlobalIndex( 2 ),
            GlobalIndex( 2 ) ),
         connectivity,
         2 };
   }
};

bool TestForwardCellBSR()
{
   static_assert(
      details::is_bsr_assembly_backend_compatible_v<
         HostBSRBackend<>,
         9,
         4 > );
   static_assert(
      details::is_bsr_assembly_backend_compatible_v<
         NativeDeviceBSRBackend<>,
         9,
         4 > );
   static_assert(
      !details::is_bsr_assembly_backend_compatible_v<
         CuSparseBSRBackend<>,
         9,
         4 > );
   static_assert(
      !details::is_bsr_assembly_backend_compatible_v<
         RocSparseBSRBackend<>,
         9,
         4 > );

   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto form = integrate( Cells< "mesh" >{}, u * v );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = RectangularKernelPolicy;

   auto full =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto element =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::BSR,
         KernelPolicy >(
            form,
            context,
            integration_rule );
   auto sgbsr_full =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto sgbsr_element =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::SGBSR,
         KernelPolicy >(
            form,
            context,
            integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

#if defined(GENDIL_USE_DEVICE)
   static_assert(
      std::is_same_v<
         typename decltype( full )::backend_type,
         NativeDeviceBSRBackend<> > );
   static_assert(
      std::is_same_v<
         typename decltype( element )::backend_type,
         NativeDeviceBSRBackend<> > );
   static_assert(
      std::is_same_v<
         typename decltype( sgbsr_full )::backend_type,
         NativeDeviceBSRBackend<> > );
#else
   static_assert(
      std::is_same_v<
         typename decltype( full )::backend_type,
         HostBSRBackend<> > );
   static_assert(
      std::is_same_v<
         typename decltype( element )::backend_type,
         HostBSRBackend<> > );
   static_assert(
      std::is_same_v<
         typename decltype( sgbsr_full )::backend_type,
         HostBSRBackend<> > );
#endif

   const auto full_data = GetHostReadView( full );
   const auto element_data = GetHostReadView( element );

   bool success = true;
   success = Check(
      full.block_rows == 9 && full.block_cols == 4 &&
      full.num_row_blocks == 2 && full.num_col_blocks == 2,
      "Forward cell BSR has the wrong rectangular block dimensions." ) &&
      success;
   success = Check(
      full.num_blocks == 4 &&
      full_data.row_offsets[0] == 0 &&
      full_data.row_offsets[1] == 2 &&
      full_data.row_offsets[2] == 4 &&
      full_data.col_indices[0] == 0 &&
      full_data.col_indices[1] == 1 &&
      full_data.col_indices[2] == 0 &&
      full_data.col_indices[3] == 1,
      "Forward full BSR adjacency is not derived from the domain mesh." ) &&
      success;
   success = Check(
      element.num_blocks == 2 &&
      element.block_rows == 9 && element.block_cols == 4 &&
      element_data.row_offsets[0] == 0 &&
      element_data.row_offsets[1] == 1 &&
      element_data.row_offsets[2] == 2 &&
      element_data.col_indices[0] == 0 &&
      element_data.col_indices[1] == 1,
      "Forward element BSR does not contain two 9 x 4 diagonal blocks." ) &&
      success;
   success = Check(
      sgbsr_full.NumRows() == 18 && sgbsr_full.NumCols() == 8 &&
      sgbsr_full.TrialBsrSize() == 8 &&
      sgbsr_full.TestBsrSize() == 18 &&
      sgbsr_full.x_bsr.Size() == 8 && sgbsr_full.y_bsr.Size() == 18 &&
      sgbsr_full.bsr_matrix.num_blocks == 2 &&
      sgbsr_full.bsr_matrix.block_rows == 9 &&
      sgbsr_full.bsr_matrix.block_cols == 4,
      "Forward cell SGBSR has the wrong external or internal dimensions." ) &&
      success;
   success = Check(
      sgbsr_element.NumRows() == 18 && sgbsr_element.NumCols() == 8 &&
      sgbsr_element.bsr_matrix.num_blocks == 2 &&
      sgbsr_element.bsr_matrix.block_rows == 9 &&
      sgbsr_element.bsr_matrix.block_cols == 4,
      "Forward element SGBSR has the wrong rectangular dimensions." ) &&
      success;

   Vector x( 8 );
   FillDeterministicInput( x );
   const auto expected = ApplyOperator( generic, x, 18 );
   success = CheckAction(
      full,
      x,
      expected,
      "Forward cell BSR action disagrees with MakeGenericOperator." ) &&
      success;
   success = CheckAction(
      element,
      x,
      expected,
      "Forward element BSR action disagrees with MakeGenericOperator." ) &&
      success;
   success = CheckAction(
      sgbsr_full,
      x,
      expected,
      "Forward cell SGBSR action disagrees with MakeGenericOperator." ) &&
      success;
   success = CheckAction(
      sgbsr_element,
      x,
      expected,
      "Forward element SGBSR action disagrees with MakeGenericOperator." ) &&
      success;
   success = CheckSGBSRDirectPipeline(
      sgbsr_full,
      x,
      expected,
      "Forward cell SGBSR disagrees with gather -> BSR -> scatter." ) &&
      success;

   Vector repeated_result( sgbsr_full.NumRows() );
   sgbsr_full( x, repeated_result );
   const Real * x_workspace = WorkspacePointer( sgbsr_full.x_bsr );
   const Real * y_workspace = WorkspacePointer( sgbsr_full.y_bsr );
   sgbsr_full( x, repeated_result );
   success = Check(
      WorkspacePointer( sgbsr_full.x_bsr ) == x_workspace &&
      WorkspacePointer( sgbsr_full.y_bsr ) == y_workspace &&
      sgbsr_full.x_bsr.Size() == 8 && sgbsr_full.y_bsr.Size() == 18,
      "Repeated rectangular SGBSR application reallocated a workspace." ) &&
      success;
#if defined(GENDIL_USE_DEVICE)
   success = Check(
      sgbsr_full.x_bsr.IsDeviceValid() &&
      sgbsr_full.y_bsr.IsDeviceValid() &&
      repeated_result.IsDeviceValid() &&
      !sgbsr_full.x_bsr.IsHostValid() &&
      !sgbsr_full.y_bsr.IsHostValid() &&
      !repeated_result.IsHostValid(),
      "Device rectangular SGBSR application staged an internal workspace or "
      "output through host memory." ) &&
      success;
#endif

   repeated_result = 1.25;
   ApplyAdd( sgbsr_full, x, repeated_result );
   const Real * repeated_data = repeated_result.ReadHostData();
   const Real * expected_data = expected.ReadHostData();
   for ( GlobalIndex i = 0; i < expected.Size(); ++i )
   {
      success = Check(
         Near( repeated_data[i], expected_data[i] + 1.25 ),
         "Rectangular SGBSR ApplyAdd did not preserve the initial output." ) &&
         success;
   }

   auto moved_sgbsr = std::move( sgbsr_full );
   success = Check(
      moved_sgbsr.NumRows() == 18 && moved_sgbsr.NumCols() == 8 &&
      moved_sgbsr.x_bsr.Size() == 8 && moved_sgbsr.y_bsr.Size() == 18,
      "Rectangular SGBSR move construction lost dimensions or workspaces." ) &&
      success;
   success = CheckAction(
      moved_sgbsr,
      x,
      expected,
      "Moved rectangular SGBSR action is incorrect." ) && success;

   auto move_assignment_source =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto move_assignment_target =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   move_assignment_target = std::move( move_assignment_source );
   success = Check(
      move_assignment_target.NumRows() == 18 &&
      move_assignment_target.NumCols() == 8 &&
      move_assignment_target.x_bsr.Size() == 8 &&
      move_assignment_target.y_bsr.Size() == 18,
      "Rectangular SGBSR move assignment lost dimensions or workspaces." ) &&
      success;
   success = CheckAction(
      move_assignment_target,
      x,
      expected,
      "Move-assigned rectangular SGBSR action is incorrect." ) && success;

   return success;
}

bool TestForwardAffineBoundaryBSR()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto source = MakeCoefficient< "source" >(
      [] GENDIL_HOST_DEVICE () -> Real { return 1.375; } );
   auto form =
      integrate(
         BoundaryFacets< "mesh" >{},
         u * v + source * v );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = RectangularKernelPolicy;

   auto full =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto element =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::BSR,
         KernelPolicy >(
            form,
            context,
            integration_rule );
   auto sgbsr_full =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto sgbsr_element =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::SGBSR,
         KernelPolicy >(
            form,
            context,
            integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      full.block_rows == 9 && full.block_cols == 4 &&
      full.num_row_blocks == 2 && full.num_col_blocks == 2,
      "Forward affine-boundary BSR has the wrong block dimensions." ) &&
      success;
   success = Check(
      element.num_blocks == 2 &&
      element.block_rows == 9 && element.block_cols == 4,
      "Forward affine-boundary element BSR has the wrong block layout." ) &&
      success;
   success = Check(
      sgbsr_full.NumRows() == 18 && sgbsr_full.NumCols() == 8 &&
      sgbsr_full.bsr_matrix.num_blocks == 4 &&
      sgbsr_full.bsr_matrix.block_rows == 9 &&
      sgbsr_full.bsr_matrix.block_cols == 4,
      "Forward affine-boundary SGBSR has the wrong dimensions or adjacency." ) &&
      success;
   success = Check(
      sgbsr_element.NumRows() == 18 && sgbsr_element.NumCols() == 8 &&
      sgbsr_element.bsr_matrix.num_blocks == 2,
      "Forward affine-boundary element SGBSR has the wrong dimensions." ) &&
      success;

   Vector x( 8 );
   FillDeterministicInput( x );
   Vector zero( 8 );
   zero = 0.0;
   const auto fx = ApplyOperator( generic, x, 18 );
   const auto fzero = ApplyOperator( generic, zero, 18 );
   Vector expected( 18 );
   Real * expected_data = expected.WriteHostData();
   const Real * fx_data = fx.ReadHostData();
   const Real * fzero_data = fzero.ReadHostData();
   for ( GlobalIndex i = 0; i < expected.Size(); ++i )
   {
      expected_data[i] = fx_data[i] - fzero_data[i];
   }

   success = CheckAction(
      full,
      x,
      expected,
      "Forward affine-boundary BSR action disagrees with F(x) - F(0)." ) &&
      success;
   success = CheckAction(
      element,
      x,
      expected,
      "Forward affine-boundary element BSR action disagrees with F(x) - F(0)." ) &&
      success;
   success = CheckAction(
      sgbsr_full,
      x,
      expected,
      "Forward affine-boundary SGBSR action disagrees with F(x) - F(0)." ) &&
      success;
   success = CheckAction(
      sgbsr_element,
      x,
      expected,
      "Forward affine-boundary element SGBSR action disagrees with F(x) - F(0)." ) &&
      success;
   success = CheckSGBSRDirectPipeline(
      sgbsr_full,
      x,
      expected,
      "Forward affine-boundary SGBSR disagrees with its direct pipeline." ) &&
      success;

   return success;
}

bool TestReverseOrientedInteriorBSR()
{
   OrientedTwoQuadFixture fixture;
   auto mesh = fixture.MakeMesh();
   const auto fixture_face =
      mesh.GetLocalFaceInfo(
         0,
         std::integral_constant< Integer, 2 >{} );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto form =
      integrate(
         InteriorFacets< "mesh" >{},
         jump( u ) * jump( v ) );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = RectangularKernelPolicy;

   auto matrix =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto element =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::BSR,
         KernelPolicy >(
            form,
            context,
            integration_rule );
   auto sgbsr_matrix =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto sgbsr_element =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::SGBSR,
         KernelPolicy >(
            form,
            context,
            integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );
   const auto matrix_data = GetHostReadView( matrix );

   bool success = true;
   success = Check(
      fixture_face.PlusSide().orientation ==
         Permutation< 2 >{ { -1, -2 } },
      "Two-quad BSR fixture did not retain its nonidentity face orientation." ) &&
      success;
   success = Check(
      matrix.block_rows == 4 && matrix.block_cols == 9 &&
      matrix.num_row_blocks == 2 && matrix.num_col_blocks == 2 &&
      matrix.num_blocks == 4,
      "Reverse-interior BSR has the wrong rectangular block pattern." ) &&
      success;
   success = Check(
      matrix_data.row_offsets[0] == 0 &&
      matrix_data.row_offsets[1] == 2 &&
      matrix_data.row_offsets[2] == 4 &&
      matrix_data.col_indices[0] == 0 &&
      matrix_data.col_indices[1] == 1 &&
      matrix_data.col_indices[2] == 0 &&
      matrix_data.col_indices[3] == 1,
      "Reverse-interior BSR adjacency is not derived from the domain mesh." ) &&
      success;
   success = Check(
      sgbsr_matrix.NumRows() == 8 && sgbsr_matrix.NumCols() == 18 &&
      sgbsr_matrix.TrialBsrSize() == 18 &&
      sgbsr_matrix.TestBsrSize() == 8 &&
      sgbsr_matrix.bsr_matrix.block_rows == 4 &&
      sgbsr_matrix.bsr_matrix.block_cols == 9 &&
      sgbsr_matrix.bsr_matrix.num_blocks == 4,
      "Reverse-interior SGBSR has the wrong external or internal dimensions." ) &&
      success;
   success = Check(
      sgbsr_element.NumRows() == 8 && sgbsr_element.NumCols() == 18 &&
      sgbsr_element.bsr_matrix.num_blocks == 2,
      "Reverse-interior element SGBSR has the wrong block pattern." ) &&
      success;

   Vector x( 18 );
   FillDeterministicInput( x );
   const auto expected = ApplyOperator( generic, x, 8 );
   success = CheckAction(
      matrix,
      x,
      expected,
      "Reverse-interior BSR action disagrees with MakeGenericOperator." ) &&
      success;
   success = CheckAction(
      sgbsr_matrix,
      x,
      expected,
      "Reverse-interior SGBSR action disagrees with MakeGenericOperator." ) &&
      success;
   success = CheckSGBSRDirectPipeline(
      sgbsr_matrix,
      x,
      expected,
      "Reverse-interior SGBSR disagrees with its oriented direct pipeline." ) &&
      success;

   const auto expected_element = ApplyOperator( element, x, 8 );
   success = CheckAction(
      sgbsr_element,
      x,
      expected_element,
      "Reverse-interior element SGBSR disagrees with diagonal BSR semantics." ) &&
      success;

   return success;
}

template < typename TrialFESpace, typename TestFESpace >
bool RunScalarH1RectangularCellCase(
   const Cartesian1DMesh & mesh,
   const TrialFESpace & trial_space,
   const TestFESpace & test_space,
   const GlobalIndex expected_cols,
   const GlobalIndex expected_rows,
   const GlobalIndex expected_block_cols,
   const GlobalIndex expected_block_rows,
   const char * dimension_message,
   const char * action_message )
{
   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto form = integrate( Cells< "mesh" >{}, u * v );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4 >{} );
   using KernelPolicy = SerialKernelConfiguration;

   auto sgbsr_full =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         form,
         context,
         integration_rule,
         HostBSRBackend<>{} );
   auto sgbsr_element =
      GenericElementBlockDiagonalAssembly<
         MatrixAssemblyType::SGBSR,
         KernelPolicy >(
            form,
            context,
            integration_rule,
            HostBSRBackend<>{} );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      sgbsr_full.NumCols() == expected_cols &&
      sgbsr_full.NumRows() == expected_rows &&
      sgbsr_full.TrialBsrSize() == 2 * expected_block_cols &&
      sgbsr_full.TestBsrSize() == 2 * expected_block_rows &&
      sgbsr_full.x_bsr.Size() == 2 * expected_block_cols &&
      sgbsr_full.y_bsr.Size() == 2 * expected_block_rows &&
      sgbsr_full.bsr_matrix.num_blocks == 2 &&
      sgbsr_full.bsr_matrix.block_cols == expected_block_cols &&
      sgbsr_full.bsr_matrix.block_rows == expected_block_rows,
      dimension_message ) && success;
   success = Check(
      sgbsr_element.NumCols() == expected_cols &&
      sgbsr_element.NumRows() == expected_rows &&
      sgbsr_element.bsr_matrix.num_blocks == 2 &&
      sgbsr_element.bsr_matrix.block_cols == expected_block_cols &&
      sgbsr_element.bsr_matrix.block_rows == expected_block_rows,
      dimension_message ) && success;

   Vector x( expected_cols );
   FillDeterministicInput( x );
   const auto expected = ApplyOperator( generic, x, expected_rows );
   success = CheckAction(
      sgbsr_full,
      x,
      expected,
      action_message ) && success;
   success = CheckAction(
      sgbsr_element,
      x,
      expected,
      action_message ) && success;
   success = CheckSGBSRDirectPipeline(
      sgbsr_full,
      x,
      expected,
      action_message ) && success;
   return success;
}

bool TestScalarH1RectangularCellSGBSR()
{
   Cartesian1DMesh mesh( 0.5, 2 );
   auto p1_fe =
      MakeLobattoFiniteElement( FiniteElementOrders< 1 >{} );
   auto p2_fe =
      MakeLobattoFiniteElement( FiniteElementOrders< 2 >{} );

   const std::array< int, 4 > p1_map{ 0, 1, 1, 2 };
   const std::array< int, 6 > p2_map{ 0, 1, 2, 2, 3, 4 };
   HostDevicePointer< const int > p1_indices{};
   HostDevicePointer< const int > p2_indices{};
   p1_indices.host_pointer = p1_map.data();
   p2_indices.host_pointer = p2_map.data();

   auto p1_space = MakeFiniteElementSpace(
      mesh,
      p1_fe,
      H1Restriction{ p1_indices, 3 } );
   auto p2_space = MakeFiniteElementSpace(
      mesh,
      p2_fe,
      H1Restriction{ p2_indices, 5 } );

   bool success = RunScalarH1RectangularCellCase(
      mesh,
      p1_space,
      p2_space,
      3,
      5,
      2,
      3,
      "Scalar H1 p1 -> p2 SGBSR dimensions are incorrect.",
      "Scalar H1 p1 -> p2 SGBSR action is incorrect." );
   success = RunScalarH1RectangularCellCase(
      mesh,
      p2_space,
      p1_space,
      5,
      3,
      3,
      2,
      "Scalar H1 p2 -> p1 SGBSR dimensions are incorrect.",
      "Scalar H1 p2 -> p1 SGBSR action is incorrect." ) && success;

   Vector p2_element_values( 6 );
   p2_element_values = 1.0;
   Vector p2_global_values( 5 );
   CGScatterFromBsr< decltype( p2_space ) >{ p2_space }(
      HostBSRBackend<>{},
      p2_element_values,
      p2_global_values );
   const Real * p2_global_data = p2_global_values.ReadHostData();
   success = Check(
      Near( p2_global_data[0], 1.0 ) &&
      Near( p2_global_data[1], 1.0 ) &&
      Near( p2_global_data[2], 2.0 ) &&
      Near( p2_global_data[3], 1.0 ) &&
      Near( p2_global_data[4], 1.0 ),
      "Scalar H1 p2 scatter did not accumulate the shared cell endpoint." ) &&
      success;

   Vector p1_element_values( 4 );
   p1_element_values = 1.0;
   Vector p1_global_values( 3 );
   CGScatterFromBsr< decltype( p1_space ) >{ p1_space }(
      HostBSRBackend<>{},
      p1_element_values,
      p1_global_values );
   const Real * p1_global_data = p1_global_values.ReadHostData();
   success = Check(
      Near( p1_global_data[0], 1.0 ) &&
      Near( p1_global_data[1], 2.0 ) &&
      Near( p1_global_data[2], 1.0 ),
      "Scalar H1 p1 scatter did not accumulate the shared cell endpoint." ) &&
      success;
   return success;
}

bool TestVectorL2ReverseOrientedInteriorSGBSR()
{
   OrientedTwoQuadFixture fixture;
   auto mesh = fixture.MakeMesh();
   auto trial_scalar =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto test_scalar =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto trial_fe =
      MakeVectorFiniteElement( trial_scalar, trial_scalar );
   auto test_fe =
      MakeVectorFiniteElement( test_scalar, test_scalar );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   VectorTrialSpace< "u" > u;
   VectorTestSpace< "v" > v;
   auto form =
      integrate(
         InteriorFacets< "mesh" >{},
         dot( jump( u ), jump( v ) ) );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = RectangularKernelPolicy;

   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = Check(
      sgbsr.NumRows() == 16 && sgbsr.NumCols() == 36 &&
      sgbsr.TestBsrSize() == 16 && sgbsr.TrialBsrSize() == 36 &&
      sgbsr.bsr_matrix.block_rows == 8 &&
      sgbsr.bsr_matrix.block_cols == 18 &&
      sgbsr.bsr_matrix.num_blocks == 4,
      "Vector L2 reverse-oriented SGBSR dimensions are incorrect." );

   Vector x( sgbsr.NumCols() );
   FillDeterministicInput( x );
   const auto expected = ApplyOperator( generic, x, sgbsr.NumRows() );
   success = CheckAction(
      sgbsr,
      x,
      expected,
      "Vector L2 reverse-oriented SGBSR action is incorrect." ) && success;
   success = CheckSGBSRDirectPipeline(
      sgbsr,
      x,
      expected,
      "Vector L2 reverse-oriented SGBSR direct pipeline is incorrect." ) &&
      success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestForwardCellBSR() && success;
   success = TestForwardAffineBoundaryBSR() && success;
#if !defined(GENDIL_USE_DEVICE)
   // The hand-built oriented unstructured fixture owns host-only geometry
   // storage. Device execution is covered above with the Cartesian fixture.
   success = TestReverseOrientedInteriorBSR() && success;
#endif
   success = TestScalarH1RectangularCellSGBSR() && success;
#if !defined(GENDIL_USE_DEVICE)
   success = TestVectorL2ReverseOrientedInteriorSGBSR() && success;
#endif
   return success ? 0 : 1;
}
