// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

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

bool Near( const Real a, const Real b )
{
   return std::abs( a - b ) < tolerance;
}

void FillDeterministicInput( Vector & x )
{
   Real * data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < static_cast< GlobalIndex >( x.Size() ); ++i )
   {
      data[i] = 1.0 + 0.5 * static_cast< Real >( i );
   }
}

void FillSentinel( Vector & y )
{
   Real * data = y.WriteHostData();
   for ( GlobalIndex i = 0; i < static_cast< GlobalIndex >( y.Size() ); ++i )
   {
      data[i] = -100.0 - static_cast< Real >( i );
   }
}

bool CheckVectorNear(
   const Vector & actual,
   const Vector & expected,
   const char * message )
{
   bool success = true;
   const Real * actual_data = actual.ReadHostData();
   const Real * expected_data = expected.ReadHostData();

   success = Check(
      actual.Size() == expected.Size(),
      message ) && success;

   const GlobalIndex size =
      static_cast< GlobalIndex >(
         actual.Size() < expected.Size() ? actual.Size() : expected.Size() );
   for ( GlobalIndex i = 0; i < size; ++i )
   {
      success = Check(
         Near( actual_data[i], expected_data[i] ),
         message ) && success;
   }
   return success;
}

template < typename Matrix >
bool CheckCSCStructure(
   const Matrix & matrix,
   const std::vector< GlobalIndex > & col_ptr,
   const std::vector< GlobalIndex > & row_ind,
   const std::vector< Real > & values,
   const char * message )
{
   bool success = true;
   success = Check(
      col_ptr.size() == static_cast< size_t >( matrix.num_cols + 1 ),
      message ) && success;
   success = Check(
      row_ind.size() == static_cast< size_t >( matrix.nnz ),
      message ) && success;
   success = Check(
      values.size() == static_cast< size_t >( matrix.nnz ),
      message ) && success;

   for ( GlobalIndex i = 0; i < matrix.num_cols + 1; ++i )
   {
      success = Check(
         matrix.col_ptr[i] == col_ptr[static_cast< size_t >( i )],
         message ) && success;
   }

   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      success = Check(
         matrix.row_ind[i] == row_ind[static_cast< size_t >( i )],
         message ) && success;
      success = Check(
         Near( matrix.values[i], values[static_cast< size_t >( i )] ),
         message ) && success;
   }
   return success;
}

template < typename Matrix >
bool CheckCSCSortedColumns( const Matrix & matrix )
{
   bool success = true;
   for ( GlobalIndex col = 0; col < matrix.num_cols; ++col )
   {
      success = Check(
         matrix.col_ptr[col] <= matrix.col_ptr[col + 1],
         "CSC col_ptr is not monotone." ) && success;

      for ( GlobalIndex entry = matrix.col_ptr[col];
            entry < matrix.col_ptr[col + 1];
            ++entry )
      {
         success = Check(
            matrix.row_ind[entry] < matrix.num_rows,
            "CSC row index is outside the matrix dimensions." ) && success;
         if ( entry + 1 < matrix.col_ptr[col + 1] )
         {
            success = Check(
               matrix.row_ind[entry] < matrix.row_ind[entry + 1],
               "CSC row indices are not strictly sorted within a column." ) && success;
         }
      }
   }
   return success;
}

void FillRawFixture( RawCOOTripletBuffer< Real, GlobalIndex > & raw )
{
   const std::array< GlobalIndex, 8 > rows{ 1, 0, 1, 0, 1, 1, 3, 0 };
   const std::array< GlobalIndex, 8 > cols{ 4, 3, 1, 3, 2, 2, 0, 0 };
   const std::array< Real, 8 > values{ 1.0, -1.0, 4.0, 2.0, 5.0, -5.0, -3.0, 2.0 };

   for ( GlobalIndex i = 0; i < raw.nnz_raw; ++i )
   {
      raw.rows[i] = rows[static_cast< size_t >( i )];
      raw.cols[i] = cols[static_cast< size_t >( i )];
      raw.values[i] = values[static_cast< size_t >( i )];
   }
}

bool TestHandWrittenCSCAction()
{
   auto matrix = MakeCSCMatrix< Real, GlobalIndex >( 4, 6, 6 );

   matrix.col_ptr[0] = 0;
   matrix.col_ptr[1] = 2;
   matrix.col_ptr[2] = 3;
   matrix.col_ptr[3] = 4;
   matrix.col_ptr[4] = 5;
   matrix.col_ptr[5] = 6;
   matrix.col_ptr[6] = 6;

   const std::array< GlobalIndex, 6 > rows{ 0, 3, 1, 1, 0, 1 };
   const std::array< Real, 6 > values{ 2.0, -3.0, 4.0, 0.0, -1.0, 1.5 };
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      matrix.row_ind[i] = rows[static_cast< size_t >( i )];
      matrix.values[i] = values[static_cast< size_t >( i )];
   }

   Vector x( 6 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 1.0;
   x_data[1] = 2.0;
   x_data[2] = 3.0;
   x_data[3] = 4.0;
   x_data[4] = 5.0;
   x_data[5] = 6.0;

   Vector y_apply( 4 );
   FillSentinel( y_apply );
   Apply( HostCSCBackend{}, matrix, x, y_apply );

   Vector expected( 4 );
   Real * expected_data = expected.WriteHostData();
   expected_data[0] = -2.0;
   expected_data[1] = 15.5;
   expected_data[2] = 0.0;
   expected_data[3] = -3.0;

   bool success = true;
   success = CheckVectorNear(
      y_apply,
      expected,
      "Apply(HostCSCBackend, CSCMatrix, x, y) produced the wrong result." ) && success;

   Vector y_operator( 4 );
   FillSentinel( y_operator );
   matrix( x, y_operator );

   success = CheckVectorNear(
      y_operator,
      expected,
      "CSCMatrix::operator()(x, y) produced the wrong result." ) && success;
   success = CheckCSCSortedColumns( matrix ) && success;

   FreeCSCMatrix( matrix );
   return success;
}

bool TestRawCOOToCSCFinalization()
{
   auto raw = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 4, 6, 8 );
   FillRawFixture( raw );

   auto csc =
      FinalizeRawCOOToCSC(
         raw,
         HostSortReduceRawCOOToCSCPolicy{} );

   bool success = true;
   success = Check(
      csc.num_rows == 4 && csc.num_cols == 6 && csc.nnz == 6,
      "RawCOO-to-CSC finalization produced the wrong dimensions." ) && success;
   success = CheckCSCStructure(
      csc,
      { 0, 2, 3, 4, 5, 6, 6 },
      { 0, 3, 1, 1, 0, 1 },
      { 2.0, -3.0, 4.0, 0.0, 1.0, 1.0 },
      "RawCOO-to-CSC finalization produced the wrong storage." ) && success;
   success = CheckCSCSortedColumns( csc ) && success;

   FreeCSCMatrix( csc );
   FreeRawCOOTripletBuffer( raw );
   return success;
}

bool TestScalarH1CSCAssemblyAgainstCSR()
{
   Cartesian1DMesh mesh( 0.5, 2 );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto fe = MakeLobattoFiniteElement( orders );

   const std::vector< int > restriction_map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction restriction{ restriction_indices, 3 };
   auto fe_space = MakeFiniteElementSpace( mesh, fe, restriction );

   Cells< "mesh" > domain;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( domain, u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< order + 2 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_csc( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_csr( fe_space.GetNumberOfFiniteElementDofs() );
   csc( x, y_csc );
   csr( x, y_csr );

   bool success = true;
   success = Check(
      csc.num_rows == csr.num_rows &&
      csc.num_cols == csr.num_cols &&
      csc.nnz == csr.nnz,
      "Scalar H1 CSC assembly dimensions disagree with CSR." ) && success;
   success = CheckCSCSortedColumns( csc ) && success;
   success = CheckVectorNear(
      y_csc,
      y_csr,
      "Scalar H1 CSC action disagrees with CSR action." ) && success;

   FreeCSRMatrix( csr );
   FreeCSCMatrix( csc );
   return success;
}

bool TestVectorH1CSCAssemblyAgainstCSR()
{
   Cartesian1DMesh mesh( 0.5, 2 );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto scalar_fe = MakeLobattoFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );

   const std::vector< int > restriction_map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe, restriction );

   Cells< "mesh" > domain;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( domain, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< order + 2 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_csc( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_csr( fe_space.GetNumberOfFiniteElementDofs() );
   csc( x, y_csc );
   csr( x, y_csr );

   bool success = true;
   success = Check(
      fe_space.GetNumberOfFiniteElementDofs() == 6,
      "Vector H1 CSC test expected component-major true DoFs." ) && success;
   success = Check(
      csc.num_rows == csr.num_rows &&
      csc.num_cols == csr.num_cols &&
      csc.nnz == csr.nnz,
      "Vector H1 CSC assembly dimensions disagree with CSR." ) && success;
   success = CheckCSCSortedColumns( csc ) && success;
   success = CheckVectorNear(
      y_csc,
      y_csr,
      "Vector H1 CSC action disagrees with CSR action." ) && success;

   FreeCSRMatrix( csr );
   FreeCSCMatrix( csc );
   return success;
}

bool TestScalarP0InteriorJumpCSCAssembly()
{
   // Analytic p0 assumptions for this mesh/form:
   // - each element basis value is 1 everywhere on the element;
   // - the single interior vertical face has physical measure 1.
   // Therefore integrate(interior_facets, jump(u) * jump(v)) on two cells is
   // [[1, -1], [-1, 1]].
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   InteriorFacets< "mesh" > interior_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( interior_facets, jump( u ) * jump( v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = CheckCSCStructure(
      csc,
      { 0, 2, 4 },
      { 0, 1, 0, 1 },
      { 1.0, -1.0, -1.0, 1.0 },
      "Analytic p0 CSC interior jump storage is wrong." ) && success;
   success = CheckCSCSortedColumns( csc ) && success;

   Vector x( 2 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 2.0;
   x_data[1] = 5.0;

   Vector y( 2 );
   csc( x, y );
   const Real * y_data = y.ReadHostData();
   success = Check( Near( y_data[0], -3.0 ), "Analytic p0 CSC action row 0 is wrong." ) && success;
   success = Check( Near( y_data[1], 3.0 ), "Analytic p0 CSC action row 1 is wrong." ) && success;

   FreeCSCMatrix( csc );
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestHandWrittenCSCAction() && success;
   success = TestRawCOOToCSCFinalization() && success;
   success = TestScalarH1CSCAssemblyAgainstCSR() && success;
   success = TestVectorH1CSCAssemblyAgainstCSR() && success;
   success = TestScalarP0InteriorJumpCSCAssembly() && success;
   return success ? 0 : 1;
}
