// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
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
bool CheckCSRStructure(
   const Matrix & matrix,
   const std::vector< GlobalIndex > & row_ptr,
   const std::vector< GlobalIndex > & col_ind,
   const std::vector< Real > & values,
   const char * message )
{
   bool success = true;
   success = Check(
      row_ptr.size() == static_cast< size_t >( matrix.num_rows + 1 ),
      message ) && success;
   success = Check(
      col_ind.size() == static_cast< size_t >( matrix.nnz ),
      message ) && success;
   success = Check(
      values.size() == static_cast< size_t >( matrix.nnz ),
      message ) && success;

   for ( GlobalIndex i = 0; i < matrix.num_rows + 1; ++i )
   {
      success = Check(
         matrix.row_ptr[i] == row_ptr[static_cast< size_t >( i )],
         message ) && success;
   }

   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      success = Check(
         matrix.col_ind[i] == col_ind[static_cast< size_t >( i )],
         message ) && success;
      success = Check(
         Near( matrix.values[i], values[static_cast< size_t >( i )] ),
         message ) && success;
   }
   return success;
}

template < typename Matrix >
bool CheckCSRSortedRows( const Matrix & matrix )
{
   bool success = true;
   for ( GlobalIndex row = 0; row < matrix.num_rows; ++row )
   {
      success = Check(
         matrix.row_ptr[row] <= matrix.row_ptr[row + 1],
         "CSR row_ptr is not monotone." ) && success;

      for ( GlobalIndex entry = matrix.row_ptr[row];
            entry < matrix.row_ptr[row + 1];
            ++entry )
      {
         success = Check(
            matrix.col_ind[entry] < matrix.num_cols,
            "CSR column index is outside the matrix dimensions." ) && success;
         if ( entry + 1 < matrix.row_ptr[row + 1] )
         {
            success = Check(
               matrix.col_ind[entry] < matrix.col_ind[entry + 1],
               "CSR column indices are not strictly sorted within a row." ) && success;
         }
      }
   }
   return success;
}

bool CheckTriplets(
   const std::vector< details::RawCOOTriplet< Real, GlobalIndex > > & triplets,
   const std::vector< GlobalIndex > & rows,
   const std::vector< GlobalIndex > & cols,
   const std::vector< Real > & values,
   const char * message )
{
   bool success = true;
   success = Check( triplets.size() == rows.size(), message ) && success;
   success = Check( triplets.size() == cols.size(), message ) && success;
   success = Check( triplets.size() == values.size(), message ) && success;

   const size_t size =
      std::min( triplets.size(), std::min( rows.size(), cols.size() ) );
   for ( size_t i = 0; i < size; ++i )
   {
      success = Check( triplets[i].row == rows[i], message ) && success;
      success = Check( triplets[i].col == cols[i], message ) && success;
      success = Check( Near( triplets[i].value, values[i] ), message ) && success;
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

bool TestHandWrittenCSRAction()
{
   auto matrix = MakeCSRMatrix< Real, GlobalIndex >( 4, 5, 6 );

   matrix.row_ptr[0] = 0;
   matrix.row_ptr[1] = 2;
   matrix.row_ptr[2] = 5;
   matrix.row_ptr[3] = 5;
   matrix.row_ptr[4] = 6;

   const std::array< GlobalIndex, 6 > cols{ 0, 3, 1, 2, 4, 0 };
   const std::array< Real, 6 > values{ 2.0, -1.0, 4.0, 0.0, 1.5, -3.0 };
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      matrix.col_ind[i] = cols[static_cast< size_t >( i )];
      matrix.values[i] = values[static_cast< size_t >( i )];
   }

   Vector x( 5 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 1.0;
   x_data[1] = 2.0;
   x_data[2] = 3.0;
   x_data[3] = 4.0;
   x_data[4] = 5.0;

   Vector y_apply( 4 );
   FillSentinel( y_apply );
   Apply( HostCSRBackend{}, matrix, x, y_apply );

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
      "Apply(HostCSRBackend, CSRMatrix, x, y) produced the wrong result." ) && success;

   Vector y_operator( 4 );
   FillSentinel( y_operator );
   matrix( x, y_operator );

   success = CheckVectorNear(
      y_operator,
      expected,
      "CSRMatrix::operator()(x, y) produced the wrong result." ) && success;
   success = CheckCSRSortedRows( matrix ) && success;

   FreeCSRMatrix( matrix );
   return success;
}

bool TestRawCOOSortReduceOrdersAndCSRFinalization()
{
   auto raw = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 4, 5, 8 );
   FillRawFixture( raw );

   const auto row_major =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( raw );
   const auto col_major =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::ColumnThenRow >( raw );

   bool success = true;
   success = CheckTriplets(
      row_major,
      { 0, 0, 1, 1, 1, 3 },
      { 0, 3, 1, 2, 4, 0 },
      { 2.0, 1.0, 4.0, 0.0, 1.0, -3.0 },
      "RawCOO row-major sort/reduce produced the wrong triplets." ) && success;
   success = CheckTriplets(
      col_major,
      { 0, 3, 1, 1, 0, 1 },
      { 0, 0, 1, 2, 3, 4 },
      { 2.0, -3.0, 4.0, 0.0, 1.0, 1.0 },
      "RawCOO column-major sort/reduce produced the wrong triplets." ) && success;

   auto csr =
      FinalizeRawCOOToCSR(
         raw,
         HostSortReduceRawCOOToCSRPolicy{} );

   success = Check(
      csr.num_rows == 4 && csr.num_cols == 5 && csr.nnz == 6,
      "RawCOO-to-CSR finalization produced the wrong dimensions." ) && success;
   success = CheckCSRStructure(
      csr,
      { 0, 2, 5, 5, 6 },
      { 0, 3, 1, 2, 4, 0 },
      { 2.0, 1.0, 4.0, 0.0, 1.0, -3.0 },
      "RawCOO-to-CSR finalization produced the wrong storage." ) && success;
   success = CheckCSRSortedRows( csr ) && success;

   FreeCSRMatrix( csr );
   FreeRawCOOTripletBuffer( raw );
   return success;
}

bool TestScalarH1CSRAssemblyAgainstCOO()
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

   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_csr( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   csr( x, y_csr );
   coo( x, y_coo );

   bool success = true;
   success = Check(
      csr.num_rows == coo.num_rows &&
      csr.num_cols == coo.num_cols &&
      csr.nnz == coo.nnz,
      "Scalar H1 CSR assembly dimensions disagree with COO." ) && success;
   success = CheckCSRSortedRows( csr ) && success;
   success = CheckVectorNear(
      y_csr,
      y_coo,
      "Scalar H1 CSR action disagrees with COO action." ) && success;

   FreeCOOMatrix( coo );
   FreeCSRMatrix( csr );
   return success;
}

bool TestVectorH1CSRAssemblyAgainstCOO()
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

   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_csr( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   csr( x, y_csr );
   coo( x, y_coo );

   bool success = true;
   success = Check(
      fe_space.GetNumberOfFiniteElementDofs() == 6,
      "Vector H1 CSR test expected component-major true DoFs." ) && success;
   success = Check(
      csr.num_rows == coo.num_rows &&
      csr.num_cols == coo.num_cols &&
      csr.nnz == coo.nnz,
      "Vector H1 CSR assembly dimensions disagree with COO." ) && success;
   success = CheckCSRSortedRows( csr ) && success;
   success = CheckVectorNear(
      y_csr,
      y_coo,
      "Vector H1 CSR action disagrees with COO action." ) && success;

   FreeCOOMatrix( coo );
   FreeCSRMatrix( csr );
   return success;
}

bool TestScalarP0InteriorJumpCSRAssembly()
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

   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = CheckCSRStructure(
      csr,
      { 0, 2, 4 },
      { 0, 1, 0, 1 },
      { 1.0, -1.0, -1.0, 1.0 },
      "Analytic p0 CSR interior jump storage is wrong." ) && success;
   success = CheckCSRSortedRows( csr ) && success;

   Vector x( 2 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 2.0;
   x_data[1] = 5.0;

   Vector y( 2 );
   csr( x, y );
   const Real * y_data = y.ReadHostData();
   success = Check( Near( y_data[0], -3.0 ), "Analytic p0 CSR action row 0 is wrong." ) && success;
   success = Check( Near( y_data[1], 3.0 ), "Analytic p0 CSR action row 1 is wrong." ) && success;

   FreeCSRMatrix( csr );
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestHandWrittenCSRAction() && success;
   success = TestRawCOOSortReduceOrdersAndCSRFinalization() && success;
   success = TestScalarH1CSRAssemblyAgainstCOO() && success;
   success = TestVectorH1CSRAssemblyAgainstCOO() && success;
   success = TestScalarP0InteriorJumpCSRAssembly() && success;
   return success ? 0 : 1;
}
