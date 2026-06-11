// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-6;

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

void FillInput( Vector & x )
{
   Real * data = x.WriteHostData();
   data[0] = 1.0;
   data[1] = 2.0;
   data[2] = 3.0;
   data[3] = 4.0;
}

void FillSentinel( Vector & y )
{
   Real * data = y.WriteHostData();
   for ( GlobalIndex i = 0; i < static_cast< GlobalIndex >( y.Size() ); ++i )
   {
      data[i] = -100.0 - static_cast< Real >( i );
   }
}

bool CheckExpected( const Vector & y, const char * message )
{
   const Real expected[] = { 5.0, 11.0, -1.0, -2.0 };
   const Real * data = y.ReadHostData();
   bool success = true;
   for ( GlobalIndex i = 0; i < 4; ++i )
   {
      success = Check( Near( data[i], expected[i] ), message ) && success;
   }
   return success;
}

bool CheckArithmeticTraits()
{
   using VectorInputValue =
      SparsePointerValueType_t<
         decltype( std::declval< const Vector & >().ReadHostData() ) >;

   static_assert( std::is_same_v< VectorInputValue, Real > );
   static_assert(
      std::is_same_v<
         DefaultSparseComputeType_t< float, VectorInputValue >,
         std::common_type_t< float, VectorInputValue > > );
   static_assert(
      std::is_same_v<
         ResolveSparseComputeType_t<
            HostCSRBackend<>,
            float,
            VectorInputValue >,
         std::common_type_t< float, VectorInputValue > > );
   static_assert(
      std::is_same_v<
         ResolveSparseAccumulatorType_t<
            HostCSRBackend<>,
            std::common_type_t< float, VectorInputValue > >,
         std::common_type_t< float, VectorInputValue > > );
   static_assert(
      std::is_same_v<
         ResolveSparseComputeType_t<
            HostCSRBackend< float, float >,
            float,
            VectorInputValue >,
         float > );
   static_assert(
      std::is_same_v<
         ResolveSparseAccumulatorType_t<
            HostCSRBackend< float, float >,
            float >,
         float > );
   static_assert(
      std::is_same_v<
         ResolveSparseComputeType_t<
            HostCOOBackend< float >,
            float,
            VectorInputValue >,
         float > );

   return true;
}

bool TestFloatBSRApplyArithmetic()
{
   auto matrix =
      MakeBlockDiagonalDGBSRPattern<
         float,
         GlobalIndex,
         BlockLayout::ColumnMajor,
         HostBSRBackend< float, float > >(
            2,
            2,
            2,
            HostBSRBackend< float, float >{} );

   using MatrixType = std::remove_cvref_t< decltype( matrix ) >;
   static_assert( std::is_same_v< typename MatrixType::value_type, float > );
   static_assert(
      std::is_same_v<
         typename MatrixType::backend_type,
         HostBSRBackend< float, float > > );

   matrix.values[0] = 1.0f;
   matrix.values[1] = 3.0f;
   matrix.values[2] = 2.0f;
   matrix.values[3] = 4.0f;
   matrix.values[4] = -1.0f;
   matrix.values[5] = 2.0f;
   matrix.values[6] = 0.5f;
   matrix.values[7] = -2.0f;

   Vector x( 4 );
   FillInput( x );

   Vector y_apply( 4 );
   FillSentinel( y_apply );
   Apply( HostBSRBackend< float, float >{}, matrix, x, y_apply );

   Vector y_operator( 4 );
   FillSentinel( y_operator );
   matrix( x, y_operator );

   bool success = true;
   success = CheckExpected(
      y_apply,
      "Float BSR Apply with compute/accumulator override failed." ) && success;
   success = CheckExpected(
      y_operator,
      "Float BSR operator() with stored backend override failed." ) && success;
   return success;
}

bool TestFloatCOOApplyArithmetic()
{
   auto matrix =
      MakeCOOMatrix< float, GlobalIndex, HostCOOBackend< float > >(
         4,
         4,
         8,
         HostCOOBackend< float >{} );

   using MatrixType = std::remove_cvref_t< decltype( matrix ) >;
   static_assert( std::is_same_v< typename MatrixType::value_type, float > );
   static_assert(
      std::is_same_v<
         typename MatrixType::backend_type,
         HostCOOBackend< float > > );

   const GlobalIndex rows[] = { 0, 0, 1, 1, 2, 2, 3, 3 };
   const GlobalIndex cols[] = { 0, 1, 0, 1, 2, 3, 2, 3 };
   const float values[] = { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -2.0f };
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      matrix.rows[i] = rows[i];
      matrix.cols[i] = cols[i];
      matrix.values[i] = values[i];
   }

   Vector x( 4 );
   FillInput( x );

   Vector y_apply( 4 );
   FillSentinel( y_apply );
   Apply( HostCOOBackend< float >{}, matrix, x, y_apply );

   Vector y_operator( 4 );
   FillSentinel( y_operator );
   matrix( x, y_operator );

   bool success = true;
   success = CheckExpected(
      y_apply,
      "Float COO Apply with compute override failed." ) && success;
   success = CheckExpected(
      y_operator,
      "Float COO operator() with stored compute override failed." ) && success;

   FreeCOOMatrix( matrix );
   return success;
}

bool TestFloatCSRApplyArithmetic()
{
   auto matrix =
      MakeCSRMatrix< float, GlobalIndex, HostCSRBackend< float, float > >(
         4,
         4,
         8,
         HostCSRBackend< float, float >{} );

   using MatrixType = std::remove_cvref_t< decltype( matrix ) >;
   static_assert( std::is_same_v< typename MatrixType::value_type, float > );
   static_assert(
      std::is_same_v<
         typename MatrixType::backend_type,
         HostCSRBackend< float, float > > );

   const GlobalIndex row_ptr[] = { 0, 2, 4, 6, 8 };
   const GlobalIndex cols[] = { 0, 1, 0, 1, 2, 3, 2, 3 };
   const float values[] = { 1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -2.0f };
   for ( GlobalIndex i = 0; i < 5; ++i )
   {
      matrix.row_ptr[i] = row_ptr[i];
   }
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      matrix.col_ind[i] = cols[i];
      matrix.values[i] = values[i];
   }

   Vector x( 4 );
   FillInput( x );

   Vector y_apply( 4 );
   FillSentinel( y_apply );
   Apply( HostCSRBackend< float, float >{}, matrix, x, y_apply );

   Vector y_operator( 4 );
   FillSentinel( y_operator );
   matrix( x, y_operator );

   bool success = true;
   success = CheckExpected(
      y_apply,
      "Float CSR Apply with compute/accumulator override failed." ) && success;
   success = CheckExpected(
      y_operator,
      "Float CSR operator() with stored backend override failed." ) && success;

   FreeCSRMatrix( matrix );
   return success;
}

bool TestFloatCSCApplyArithmetic()
{
   auto matrix =
      MakeCSCMatrix< float, GlobalIndex, HostCSCBackend< float > >(
         4,
         4,
         8,
         HostCSCBackend< float >{} );

   using MatrixType = std::remove_cvref_t< decltype( matrix ) >;
   static_assert( std::is_same_v< typename MatrixType::value_type, float > );
   static_assert(
      std::is_same_v<
         typename MatrixType::backend_type,
         HostCSCBackend< float > > );

   const GlobalIndex col_ptr[] = { 0, 2, 4, 6, 8 };
   const GlobalIndex rows[] = { 0, 1, 0, 1, 2, 3, 2, 3 };
   const float values[] = { 1.0f, 3.0f, 2.0f, 4.0f, -1.0f, 2.0f, 0.5f, -2.0f };
   for ( GlobalIndex i = 0; i < 5; ++i )
   {
      matrix.col_ptr[i] = col_ptr[i];
   }
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      matrix.row_ind[i] = rows[i];
      matrix.values[i] = values[i];
   }

   Vector x( 4 );
   FillInput( x );

   Vector y_apply( 4 );
   FillSentinel( y_apply );
   Apply( HostCSCBackend< float >{}, matrix, x, y_apply );

   Vector y_operator( 4 );
   FillSentinel( y_operator );
   matrix( x, y_operator );

   bool success = true;
   success = CheckExpected(
      y_apply,
      "Float CSC Apply with compute override failed." ) && success;
   success = CheckExpected(
      y_operator,
      "Float CSC operator() with stored compute override failed." ) && success;

   FreeCSCMatrix( matrix );
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = CheckArithmeticTraits() && success;
   success = TestFloatBSRApplyArithmetic() && success;
   success = TestFloatCOOApplyArithmetic() && success;
   success = TestFloatCSRApplyArithmetic() && success;
   success = TestFloatCSCApplyArithmetic() && success;
   return success ? 0 : 1;
}
