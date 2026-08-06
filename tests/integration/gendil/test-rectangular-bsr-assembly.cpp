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
   using KernelPolicy = SerialKernelConfiguration;

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
#else
   static_assert(
      std::is_same_v<
         typename decltype( full )::backend_type,
         HostBSRBackend<> > );
   static_assert(
      std::is_same_v<
         typename decltype( element )::backend_type,
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
   using KernelPolicy = SerialKernelConfiguration;

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
   using KernelPolicy = SerialKernelConfiguration;

   auto matrix =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
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

   Vector x( 18 );
   FillDeterministicInput( x );
   const auto expected = ApplyOperator( generic, x, 8 );
   success = CheckAction(
      matrix,
      x,
      expected,
      "Reverse-interior BSR action disagrees with MakeGenericOperator." ) &&
      success;

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestForwardCellBSR() && success;
   success = TestForwardAffineBoundaryBSR() && success;
   success = TestReverseOrientedInteriorBSR() && success;
   return success ? 0 : 1;
}
