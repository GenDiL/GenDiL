// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <utility>

using namespace gendil;

namespace
{

using ScalarFE0 = GLFiniteElement< 1, 1 >;
using ScalarFE1 = GLFiniteElement< 2, 1 >;
using ScalarShape0 = typename ScalarFE0::shape_functions;
using VectorFE = decltype( MakeVectorFiniteElement( ScalarFE0{}, ScalarFE1{} ) );
using VectorShape = typename VectorFE::shape_functions;
using VectorSpace = FiniteElementSpace< Cartesian2DMesh, VectorFE, L2Restriction >;
using Component0Tag = std::integral_constant< size_t, 0 >;
using Component1Tag = std::integral_constant< size_t, 1 >;

constexpr Real tolerance = 1.0e-12;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool Near( const Real a, const Real b )
{
   return std::abs( a - b ) < tolerance;
}

constexpr std::array< int, 8 > h1_q1_restriction_map{
   0, 1, 3, 4,
   1, 2, 4, 5
};

HostDevicePointer< const int > MakeManualH1RestrictionIndices()
{
   HostDevicePointer< const int > indices{};
   indices.host_pointer = h1_q1_restriction_map.data();
   return indices;
}

void FillVectorH1InputCase( Vector & x, const Integer case_id )
{
   Real * data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      data[i] = 0.0;
   }

   if ( case_id == 0 )
   {
      for ( GlobalIndex i = 0; i < x.Size(); ++i )
      {
         data[i] = 0.5 + 0.25 * static_cast< Real >( i );
      }
   }
   else if ( case_id == 1 )
   {
      data[0] = 1.0;
      data[1] = 2.0;
      data[2] = 3.0;
   }
   else
   {
      data[3] = -1.0;
      data[4] = 0.5;
      data[5] = 4.0;
   }
}

void ApplyTwoCellVectorH1P1MassReference(
   const Vector & x,
   Vector & y,
   const Real h )
{
   // Two uniform 1D p1 H1 elements have local scalar mass
   // (h / 6) * [[2, 1], [1, 2]]. Assembling the shared middle
   // node gives the component-major diag(M_scalar, M_scalar) action.
   const Real scale = h / 6.0;
   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   for ( GlobalIndex i = 0; i < y.Size(); ++i )
   {
      y_data[i] = 0.0;
   }

   for ( GlobalIndex component = 0; component < 2; ++component )
   {
      const GlobalIndex offset = 3 * component;
      y_data[offset + 0] =
         scale * ( 2.0 * x_data[offset + 0] + x_data[offset + 1] );
      y_data[offset + 1] =
         scale * ( x_data[offset + 0] +
                   4.0 * x_data[offset + 1] +
                   x_data[offset + 2] );
      y_data[offset + 2] =
         scale * ( x_data[offset + 1] + 2.0 * x_data[offset + 2] );
   }
}

bool CheckVectorNear(
   const Vector & actual,
   const Vector & expected,
   const char * message )
{
   const Real * actual_data = actual.ReadHostData();
   const Real * expected_data = expected.ReadHostData();

   bool success = true;
   for ( GlobalIndex i = 0; i < actual.Size(); ++i )
   {
      success = Check(
         Near( actual_data[i], expected_data[i] ),
         message ) && success;
   }
   return success;
}

void FillIdentityBlocks(
   BSRMatrix< Real, GlobalIndex > & matrix )
{
   const GlobalIndex block_size = matrix.block_rows;
   GENDIL_VERIFY(
      matrix.block_rows == matrix.block_cols,
      "FillIdentityBlocks requires square BSR blocks." );

   for ( GlobalIndex i = 0;
         i < matrix.num_blocks * matrix.block_rows * matrix.block_cols;
         ++i )
   {
      matrix.values[i] = 0.0;
   }

   for ( GlobalIndex block = 0; block < matrix.num_blocks; ++block )
   {
      const GlobalIndex block_offset = block * block_size * block_size;
      for ( GlobalIndex local = 0; local < block_size; ++local )
      {
         matrix.values[block_offset + local * block_size + local] = 1.0;
      }
   }
}

bool TestIdentityWrapperMatchesRawBsr()
{
   auto raw_matrix = MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 2, 2, 3 );

   for ( GlobalIndex i = 0;
         i < raw_matrix.num_blocks * raw_matrix.block_rows * raw_matrix.block_cols;
         ++i )
   {
      raw_matrix.values[i] = static_cast< Real >( 1 + i );
   }

   Vector x( 6 );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = 0.25 + static_cast< Real >( i );
   }

   Vector y_raw( 4 );
   Vector y_sg( 4 );
   y_raw = 0.0;
   y_sg = 0.0;

   raw_matrix( x, y_raw );

   SGBSRMatrix<
      BSRMatrix< Real, GlobalIndex >,
      IdentityBsrGather,
      IdentityBsrScatter > sg_matrix(
         raw_matrix,
         IdentityBsrGather{},
         IdentityBsrScatter{} );

   sg_matrix( x, y_sg );

   bool success = true;
   success = Check(
      sg_matrix.TrialBsrSize() == 6,
      "Identity SGBSRMatrix trial BSR size is wrong." ) && success;
   success = Check(
      sg_matrix.TestBsrSize() == 4,
      "Identity SGBSRMatrix test BSR size is wrong." ) && success;

   const Real * y_raw_data = y_raw.ReadHostData();
   const Real * y_sg_data = y_sg.ReadHostData();
   for ( GlobalIndex i = 0; i < y_raw.Size(); ++i )
   {
      success = Check(
         Near( y_raw_data[i], y_sg_data[i] ),
         "Identity SGBSRMatrix apply disagrees with raw BSRMatrix." ) && success;
   }

   return success;
}

bool TestRawBsrOperatorDelegatesToBackendApply()
{
   auto raw_matrix = MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 2, 2, 3 );

   for ( GlobalIndex i = 0;
         i < raw_matrix.num_blocks * raw_matrix.block_rows * raw_matrix.block_cols;
         ++i )
   {
      raw_matrix.values[i] = static_cast< Real >( 2 + 3 * i );
   }

   Vector x( 6 );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = -0.5 + static_cast< Real >( i );
   }

   Vector y_operator( 4 );
   Vector y_apply( 4 );
   y_operator = 0.0;
   y_apply = 0.0;

   raw_matrix( x, y_operator );
   Apply( raw_matrix.backend, raw_matrix, x, y_apply );

   bool success = true;
   const Real * y_operator_data = y_operator.ReadHostData();
   const Real * y_apply_data = y_apply.ReadHostData();
   for ( GlobalIndex i = 0; i < y_operator.Size(); ++i )
   {
      success = Check(
         Near( y_operator_data[i], y_apply_data[i] ),
         "Raw BSR operator() disagrees with backend Apply." ) && success;
   }

   return success;
}

bool TestScalarH1GatherScatterMapping()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   H1Restriction restriction{ MakeManualH1RestrictionIndices(), 6 };
   auto h1_space = MakeFiniteElementSpace( mesh, ScalarFE0{}, restriction );

   constexpr GlobalIndex block_size = LocalDofCount< ScalarShape0 >();
   const GlobalIndex num_elements = h1_space.GetNumberOfFiniteElements();

   Vector x_fe( h1_space.GetNumberOfFiniteElementDofs() );
   Real * x_fe_data = x_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      x_fe_data[i] = 10.0 * static_cast< Real >( i + 1 );
   }

   CGGatherToBsr< decltype( h1_space ) > gather{ h1_space };
   Vector x_bsr( num_elements * block_size );
   gather( x_fe, x_bsr );

   bool success = true;
   const Real expected_gather[] = {
      10.0, 20.0, 40.0, 50.0,
      20.0, 30.0, 50.0, 60.0
   };
   const Real * x_bsr_data = x_bsr.ReadHostData();
   for ( GlobalIndex i = 0; i < x_bsr.Size(); ++i )
   {
      success = Check(
         Near( x_bsr_data[i], expected_gather[i] ),
         "Scalar H1 gather mapping is wrong." ) && success;
   }

   Vector y_bsr( num_elements * block_size );
   Real * y_bsr_data = y_bsr.WriteHostData();
   for ( GlobalIndex i = 0; i < y_bsr.Size(); ++i )
   {
      y_bsr_data[i] = static_cast< Real >( i + 1 );
   }

   Vector y_fe( h1_space.GetNumberOfFiniteElementDofs() );
   Real * y_fe_data = y_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      y_fe_data[i] = 99.0;
   }

   CGScatterFromBsr< decltype( h1_space ) > scatter{ h1_space };
   scatter( y_bsr, y_fe );

   const Real expected_scatter[] = {
      1.0, 7.0, 6.0, 3.0, 11.0, 8.0
   };
   y_fe_data = y_fe.ReadWriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      success = Check(
         Near( y_fe_data[i], expected_scatter[i] ),
         "Scalar H1 scatter-add mapping or Set semantics is wrong." ) && success;
   }

   return success;
}

bool TestVectorGatherScatterMapping()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, VectorFE{} );

   constexpr Component0Tag c0{};
   constexpr Component1Tag c1{};
   constexpr GlobalIndex block_size = LocalDofCount< VectorShape >();

   Vector x_fe( vector_space.GetNumberOfFiniteElementDofs() );
   Real * x_fe_data = x_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      x_fe_data[i] = -1.0;
   }

   const GlobalIndex num_elements = vector_space.GetNumberOfFiniteElements();
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< component_dof_shape_t< VectorShape, 0 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex global_index =
            GlobalDofIndex( vector_space, c0, element, indices );
         x_fe_data[global_index] =
            100.0 * element +
            static_cast< Real >( FlattenLocalDof( vector_space, c0, indices ) );
      });

      UnitLoop< component_dof_shape_t< VectorShape, 1 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex global_index =
            GlobalDofIndex( vector_space, c1, element, indices );
         x_fe_data[global_index] =
            100.0 * element +
            static_cast< Real >( FlattenLocalDof( vector_space, c1, indices ) );
      });
   }

   DGGatherToBsr< decltype( vector_space ) > gather{ vector_space };
   Vector x_bsr( num_elements * block_size );
   gather( x_fe, x_bsr );

   bool success = true;
   const Real * x_bsr_data = x_bsr.ReadHostData();
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< component_dof_shape_t< VectorShape, 0 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex bsr_index =
            element * block_size + FlattenLocalDof( vector_space, c0, indices );
         const GlobalIndex fe_index =
            GlobalDofIndex( vector_space, c0, element, indices );
         success = Check(
            Near( x_bsr_data[bsr_index], x_fe_data[fe_index] ),
            "Vector gather component 0 mapping is wrong." ) && success;
      });

      UnitLoop< component_dof_shape_t< VectorShape, 1 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex bsr_index =
            element * block_size + FlattenLocalDof( vector_space, c1, indices );
         const GlobalIndex fe_index =
            GlobalDofIndex( vector_space, c1, element, indices );
         success = Check(
            Near( x_bsr_data[bsr_index], x_fe_data[fe_index] ),
            "Vector gather component 1 mapping is wrong." ) && success;
      });
   }

   Vector y_bsr( num_elements * block_size );
   Real * y_bsr_data = y_bsr.WriteHostData();
   for ( GlobalIndex i = 0; i < y_bsr.Size(); ++i )
   {
      y_bsr_data[i] = 7.0 + static_cast< Real >( 3 * i );
   }

   DGScatterFromBsr< decltype( vector_space ) > scatter{ vector_space };
   Vector y_fe( vector_space.GetNumberOfFiniteElementDofs() );
   scatter( y_bsr, y_fe );

   const Real * y_fe_data = y_fe.ReadHostData();
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< component_dof_shape_t< VectorShape, 0 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex bsr_index =
            element * block_size + FlattenLocalDof( vector_space, c0, indices );
         const GlobalIndex fe_index =
            GlobalDofIndex( vector_space, c0, element, indices );
         success = Check(
            Near( y_fe_data[fe_index], y_bsr_data[bsr_index] ),
            "Vector scatter component 0 mapping is wrong." ) && success;
      });

      UnitLoop< component_dof_shape_t< VectorShape, 1 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex bsr_index =
            element * block_size + FlattenLocalDof( vector_space, c1, indices );
         const GlobalIndex fe_index =
            GlobalDofIndex( vector_space, c1, element, indices );
         success = Check(
            Near( y_fe_data[fe_index], y_bsr_data[bsr_index] ),
            "Vector scatter component 1 mapping is wrong." ) && success;
      });
   }

   return success;
}

bool TestVectorH1GatherScatterMapping()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto scalar_fe = MakeLobattoFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );

   const std::array< int, 4 > restriction_map{
      0, 1,
      1, 2
   };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto vector_h1_space = MakeFiniteElementSpace( mesh, vector_fe, restriction );

   using VectorH1Space = std::remove_cvref_t< decltype( vector_h1_space ) >;
   using ShapeFunctions = typename VectorH1Space::finite_element_type::shape_functions;
   constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
   const GlobalIndex num_elements =
      vector_h1_space.GetNumberOfFiniteElements();

   Vector x_fe( vector_h1_space.GetNumberOfFiniteElementDofs() );
   Real * x_fe_data = x_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      x_fe_data[i] = 10.0 * static_cast< Real >( i + 1 );
   }

   VectorCGGatherToBsr< decltype( vector_h1_space ) > gather{
      vector_h1_space };
   Vector x_bsr( num_elements * block_size );
   gather( x_fe, x_bsr );

   bool success = true;
   const Real expected_gather[] = {
      10.0, 20.0, 40.0, 50.0,
      20.0, 30.0, 50.0, 60.0
   };
   const Real * x_bsr_data = x_bsr.ReadHostData();
   for ( GlobalIndex i = 0; i < x_bsr.Size(); ++i )
   {
      success = Check(
         Near( x_bsr_data[i], expected_gather[i] ),
         "Vector H1 gather mapping is wrong." ) && success;
   }

   Vector y_bsr( num_elements * block_size );
   Real * y_bsr_data = y_bsr.WriteHostData();
   for ( GlobalIndex i = 0; i < y_bsr.Size(); ++i )
   {
      y_bsr_data[i] = static_cast< Real >( i + 1 );
   }

   Vector y_fe( vector_h1_space.GetNumberOfFiniteElementDofs() );
   Real * y_fe_data = y_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      y_fe_data[i] = 99.0;
   }

   VectorCGScatterFromBsr< decltype( vector_h1_space ) > scatter{
      vector_h1_space };
   scatter( y_bsr, y_fe );

   const Real expected_scatter[] = {
      1.0, 7.0, 6.0,
      3.0, 11.0, 8.0
   };
   y_fe_data = y_fe.ReadWriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      success = Check(
         Near( y_fe_data[i], expected_scatter[i] ),
         "Vector H1 scatter-add mapping or Set semantics is wrong." ) && success;
   }

   return success;
}

bool TestVectorSGBSRPermutationApply()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, VectorFE{} );

   constexpr GlobalIndex block_size = LocalDofCount< VectorShape >();
   auto identity_bsr =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >(
         vector_space.GetNumberOfFiniteElements(),
         block_size,
         block_size );
   FillIdentityBlocks( identity_bsr );

   Vector x_fe( vector_space.GetNumberOfFiniteElementDofs() );
   Real * x_fe_data = x_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      x_fe_data[i] = 0.5 + static_cast< Real >( i * i );
   }

   Vector y_fe( vector_space.GetNumberOfFiniteElementDofs() );

   SGBSRMatrix<
      BSRMatrix< Real, GlobalIndex >,
      DGGatherToBsr< decltype( vector_space ) >,
      DGScatterFromBsr< decltype( vector_space ) > > sg_matrix(
         identity_bsr,
         DGGatherToBsr< decltype( vector_space ) >{ vector_space },
         DGScatterFromBsr< decltype( vector_space ) >{ vector_space } );
   sg_matrix( x_fe, y_fe );

   bool success = true;
   const Real * y_fe_data = y_fe.ReadHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      success = Check(
         Near( y_fe_data[i], x_fe_data[i] ),
         "Vector SGBSR identity BSR apply did not preserve FE vector values." ) && success;
   }

   return success;
}

bool TestVectorH1SGBSRCellMass()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto scalar_fe = MakeLobattoFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );

   const std::array< int, 4 > restriction_map{
      0, 1,
      1, 2
   };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto vector_h1_space = MakeFiniteElementSpace( mesh, vector_fe, restriction );

   Cells< "mesh" > cells;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( cells, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( vector_h1_space ),
         MakeIntegrationDomain< "mesh" >( vector_h1_space ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;
   auto sgbsr_matrix =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto generic_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   for ( Integer case_id = 0; case_id < 3; ++case_id )
   {
      Vector x( vector_h1_space.GetNumberOfFiniteElementDofs() );
      Vector y_sgbsr( vector_h1_space.GetNumberOfFiniteElementDofs() );
      Vector y_operator( vector_h1_space.GetNumberOfFiniteElementDofs() );
      Vector y_expected( vector_h1_space.GetNumberOfFiniteElementDofs() );

      FillVectorH1InputCase( x, case_id );
      y_sgbsr = 0.0;
      y_operator = 0.0;

      sgbsr_matrix( x, y_sgbsr );
      generic_operator( x, y_operator );
      ApplyTwoCellVectorH1P1MassReference( x, y_expected, h );

      success = CheckVectorNear(
         y_sgbsr,
         y_expected,
         "Vector H1 SGBSR action disagrees with the manual p1 mass reference." ) && success;
      success = CheckVectorNear(
         y_sgbsr,
         y_operator,
         "Vector H1 SGBSR action disagrees with GenericOperator." ) && success;

      const Real * y_data = y_sgbsr.ReadHostData();
      const Real * expected_data = y_expected.ReadHostData();
      if ( case_id == 1 )
      {
         success = Check(
            Near( y_data[1], expected_data[1] ),
            "Vector H1 SGBSR did not accumulate the component 0 shared middle node." ) && success;
         success = Check(
            Near( y_data[3], 0.0 ) &&
            Near( y_data[4], 0.0 ) &&
            Near( y_data[5], 0.0 ),
            "Vector H1 SGBSR aliased component 0 input into component 1 output." ) && success;
      }
      else if ( case_id == 2 )
      {
         success = Check(
            Near( y_data[4], expected_data[4] ),
            "Vector H1 SGBSR did not accumulate the component 1 shared middle node." ) && success;
         success = Check(
            Near( y_data[0], 0.0 ) &&
            Near( y_data[1], 0.0 ) &&
            Near( y_data[2], 0.0 ),
            "Vector H1 SGBSR aliased component 1 input into component 0 output." ) && success;
      }
   }

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestIdentityWrapperMatchesRawBsr() && success;
   success = TestRawBsrOperatorDelegatesToBackendApply() && success;
   success = TestScalarH1GatherScatterMapping() && success;
   success = TestVectorGatherScatterMapping() && success;
   success = TestVectorH1GatherScatterMapping() && success;
   success = TestVectorSGBSRPermutationApply() && success;
   success = TestVectorH1SGBSRCellMass() && success;

   return success ? 0 : 1;
}
