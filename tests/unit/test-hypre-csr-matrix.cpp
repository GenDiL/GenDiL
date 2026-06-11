// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-12;

#ifdef GENDIL_USE_HYPRE_DEVICE
using ExpectedHypreCSRBackend = HypreCSRDeviceBackend;
#else
using ExpectedHypreCSRBackend = HypreCSRHostBackend;
#endif

static_assert(
   std::is_same_v<
      HypreCSRMatrix<>::backend_type,
      ExpectedHypreCSRBackend > );
static_assert(
   std::is_same_v<
      HypreCSRMatrix< HypreCSRHostBackend >::backend_type,
      HypreCSRHostBackend > );
static_assert(
   std::is_same_v<
      HypreCSRMatrix< HypreCSRDeviceBackend >::backend_type,
      HypreCSRDeviceBackend > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::HypreCSR >,
      ExpectedHypreCSRBackend > );

static_assert( is_host_matvec_backend_v< HostBSRBackend<> > );
static_assert( is_host_matvec_backend_v< HostCOOBackend<> > );
static_assert( is_host_matvec_backend_v< HostCSCBackend<> > );
static_assert( is_host_matvec_backend_v< HostCSRBackend<> > );
static_assert( is_host_matvec_backend_v< HypreCSRHostBackend > );
static_assert( is_device_matvec_backend_v< NativeDeviceBSRBackend<> > );
static_assert( is_device_matvec_backend_v< NativeDeviceCOOBackend<> > );
static_assert( is_device_matvec_backend_v< NativeDeviceCSCBackend<> > );
static_assert( is_device_matvec_backend_v< NativeDeviceCSRBackend<> > );
static_assert( is_device_matvec_backend_v< HypreCSRDeviceBackend > );
static_assert( !is_device_matvec_backend_v< HostCSRBackend<> > );
static_assert( !is_host_matvec_backend_v< NativeDeviceCSRBackend<> > );

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

void FillRawHypreCSRFixture(
   RawCOOTripletBuffer< Real, GlobalIndex > & raw )
{
   const std::array< GlobalIndex, 8 > rows{ 0, 0, 1, 1, 2, 0, 1, 2 };
   const std::array< GlobalIndex, 8 > cols{ 2, 0, 0, 1, 2, 0, 2, 0 };
   const std::array< Real, 8 > values{ 3.0, 1.0, -1.0, 4.0, 5.0, 1.0, 2.0, -2.0 };

   for ( GlobalIndex i = 0; i < raw.nnz_raw; ++i )
   {
      raw.rows[i] = rows[static_cast< size_t >( i )];
      raw.cols[i] = cols[static_cast< size_t >( i )];
      raw.values[i] = values[static_cast< size_t >( i )];
   }
}

template < typename Backend >
HypreCSRMatrix< Backend > BuildSmallHypreCSRMatrix( Backend backend )
{
   auto raw = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 3, 3, 8 );
   FillRawHypreCSRFixture( raw );

   auto matrix =
      FinalizeRawCOOToHypreCSR(
         raw,
         HostSortReduceRawCOOToHypreCSRPolicy{},
         backend );
   FreeRawCOOTripletBuffer( raw );
   return matrix;
}

HypreCSRMatrix<> BuildSmallHypreCSRMatrix()
{
   return BuildSmallHypreCSRMatrix( typename HypreCSRMatrix<>::backend_type{} );
}

bool CheckVector(
   const Vector & vector,
   const std::array< Real, 3 > & expected,
   const char * message )
{
   bool success = true;
   const Real * data = vector.ReadHostData();
   for ( GlobalIndex i = 0; i < expected.size(); ++i )
   {
      success = Check(
         Near( data[i], expected[static_cast< size_t >( i )] ),
         message ) && success;
   }
   return success;
}

bool TestHypreCSRFinalizationAndNativeApply()
{
   auto matrix = BuildSmallHypreCSRMatrix();

   bool success = true;
   success = Check( matrix.csr.num_rows == 3, "HypreCSR row count is wrong." ) && success;
   success = Check( matrix.csr.num_cols == 3, "HypreCSR column count is wrong." ) && success;
   success = Check( matrix.csr.nnz == 7, "HypreCSR duplicate reduction failed." ) && success;
   success = Check( matrix.metadata.is_square, "HypreCSR metadata should mark this matrix square." ) && success;
   success = Check(
      matrix.metadata.has_explicit_diagonal,
      "HypreCSR metadata should record a complete explicit diagonal." ) && success;

   const std::array< HYPRE_Int, 4 > row_ptr{ 0, 2, 5, 7 };
   const std::array< HYPRE_Int, 7 > col_ind{ 0, 2, 1, 0, 2, 2, 0 };
   const std::array< Real, 7 > values{ 2.0, 3.0, 4.0, -1.0, 2.0, 5.0, -2.0 };

   for ( HYPRE_Int i = 0; i < 4; ++i )
   {
      success = Check(
         matrix.csr.row_ptr[i] == row_ptr[static_cast< size_t >( i )],
         "HypreCSR row_ptr is wrong." ) && success;
   }

   for ( HYPRE_Int i = 0; i < 7; ++i )
   {
      success = Check(
         matrix.csr.col_ind[i] == col_ind[static_cast< size_t >( i )],
         "HypreCSR column ordering is wrong." ) && success;
      success = Check(
         Near(
            static_cast< Real >( matrix.csr.values[i] ),
            values[static_cast< size_t >( i )] ),
         "HypreCSR values are wrong." ) && success;
   }

   Vector x( 3 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 1.0;
   x_data[1] = 2.0;
   x_data[2] = 3.0;

   Vector y_operator( 3 );
   y_operator = 0.0;
   matrix( x, y_operator );

   success = CheckVector(
      y_operator,
      { 11.0, 13.0, 13.0 },
      "HypreCSRMatrix::operator()(x, y) produced the wrong Hypre-backed result." ) && success;

   Vector y_native( 3 );
   Apply( HostCSRBackend<>{}, matrix, x, y_native );

   success = CheckVector(
      y_native,
      { 11.0, 13.0, 13.0 },
      "Apply(HostCSRBackend, HypreCSRMatrix, x, y) produced the wrong native result." ) && success;

   return success;
}

bool TestHypreCSRDiagonalFirstSortReduce()
{
   auto raw = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 3, 4, 9 );
   const std::array< GlobalIndex, 9 > rows{ 1, 1, 1, 1, 1, 1, 0, 0, 2 };
   const std::array< GlobalIndex, 9 > cols{ 3, 1, 0, 3, 1, 2, 2, 2, 0 };
   const std::array< Real, 9 > values{ 1.0, 4.0, -2.0, 2.0, 5.0, 7.0, 8.0, -8.0, 9.0 };

   for ( GlobalIndex i = 0; i < raw.nnz_raw; ++i )
   {
      raw.rows[i] = rows[static_cast< size_t >( i )];
      raw.cols[i] = cols[static_cast< size_t >( i )];
      raw.values[i] = values[static_cast< size_t >( i )];
   }

   auto matrix = FinalizeRawCOOToHypreCSR( raw );
   FreeRawCOOTripletBuffer( raw );

   bool success = true;
   success = Check(
      !matrix.metadata.is_square,
      "Rectangular HypreCSR metadata should record a nonsquare matrix." ) && success;
   success = Check(
      !matrix.metadata.has_explicit_diagonal,
      "HypreCSR metadata should record missing rectangular diagonals." ) && success;
   success = Check(
      matrix.metadata.explicit_diagonal_count == HYPRE_Int( 1 ),
      "HypreCSR metadata should count the one explicit diagonal." ) && success;
   success = Check(
      matrix.metadata.missing_diagonal_count == HYPRE_Int( 2 ),
      "HypreCSR metadata should count missing eligible diagonals." ) && success;
   success = Check(
      matrix.metadata.first_missing_diagonal == HYPRE_Int( 0 ),
      "HypreCSR metadata should record the first missing diagonal." ) && success;

   const std::array< HYPRE_Int, 4 > row_ptr{ 0, 1, 5, 6 };
   const std::array< HYPRE_Int, 6 > col_ind{ 2, 1, 0, 2, 3, 0 };
   const std::array< Real, 6 > expected_values{ 0.0, 9.0, -2.0, 7.0, 3.0, 9.0 };

   success = Check( matrix.csr.nnz == 6, "HypreCSR duplicate reduction count is wrong." ) && success;
   for ( HYPRE_Int i = 0; i < 4; ++i )
   {
      success = Check(
         matrix.csr.row_ptr[i] == row_ptr[static_cast< size_t >( i )],
         "HypreCSR diagonal-first row_ptr is wrong." ) && success;
   }

   for ( HYPRE_Int i = 0; i < 6; ++i )
   {
      success = Check(
         matrix.csr.col_ind[i] == col_ind[static_cast< size_t >( i )],
         "HypreCSR diagonal-first column ordering is wrong." ) && success;
      success = Check(
         Near(
            static_cast< Real >( matrix.csr.values[i] ),
            expected_values[static_cast< size_t >( i )] ),
         "HypreCSR diagonal-first reduced values are wrong." ) && success;
   }

   return success;
}

bool TestHypreAliasVectorViewsAndMatvec()
{
   auto matrix = BuildSmallHypreCSRMatrix();
   bool success = true;

   HYPRE_ParCSRMatrix parcsr =
      matrix.GetHostHypreParCSR();
#ifndef GENDIL_USE_HYPRE_DEVICE
   success = Check(
      matrix.GetHypreParCSR() == parcsr,
      "Default Hypre ParCSR getter should return the host shell in this build." ) && success;
   success = Check(
      static_cast< HYPRE_ParCSRMatrix >( matrix ) == parcsr,
      "Hypre ParCSR conversion should return the default shell." ) && success;
#endif
   auto * internal = reinterpret_cast< hypre_ParCSRMatrix * >( parcsr );
   hypre_CSRMatrix * diag = hypre_ParCSRMatrixDiag( internal );

   success = Check(
      hypre_CSRMatrixI( diag ) == matrix.csr.row_ptr.host_pointer,
      "Hypre ParCSR row pointer does not alias GenDiL storage." ) && success;
   success = Check(
      hypre_CSRMatrixJ( diag ) == matrix.csr.col_ind.host_pointer,
      "Hypre ParCSR column pointer does not alias GenDiL storage." ) && success;
   success = Check(
      hypre_CSRMatrixData( diag ) == matrix.csr.values.host_pointer,
      "Hypre ParCSR value pointer does not alias GenDiL storage." ) && success;

   Vector x( 3 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 1.0;
   x_data[1] = 2.0;
   x_data[2] = 3.0;

   const Vector & x_const = x;
   auto x_view =
      MakeHostHypreParVectorView(
         x_const,
         HYPRE_Int( 3 ),
         matrix.metadata.comm );
   success = Check(
      x.IsHostValid(),
      "Const MakeHypreParVectorView should preserve host-valid input state." ) && success;

   Vector y( 3 );
   y = 0.0;
   auto y_view =
      MakeHostHypreParVectorView(
         y,
         HYPRE_Int( 3 ),
         matrix.metadata.comm );
   success = Check(
      y.IsHostValid() && !y.IsDeviceValid(),
      "Mutable MakeHypreParVectorView should leave host valid and device stale." ) && success;

   (void) x_view;
   (void) y_view;

   Vector y_hypre( 3 );
   y_hypre = 0.0;
   HypreMatvec( matrix, x, y_hypre );

   success = CheckVector(
      y_hypre,
      { 11.0, 13.0, 13.0 },
      "HypreMatvec produced the wrong result." ) && success;

#ifdef GENDIL_USE_HYPRE_DEVICE
   HYPRE_ParCSRMatrix device_parcsr =
      matrix.GetDeviceHypreParCSR();
   success = Check(
      matrix.GetHypreParCSR() == device_parcsr,
      "Default Hypre ParCSR getter should return the device shell in this build." ) && success;
   success = Check(
      static_cast< HYPRE_ParCSRMatrix >( matrix ) == device_parcsr,
      "Hypre ParCSR conversion should return the default shell." ) && success;
   auto * device_internal =
      reinterpret_cast< hypre_ParCSRMatrix * >( device_parcsr );
   hypre_CSRMatrix * device_diag =
      hypre_ParCSRMatrixDiag( device_internal );

   success = Check(
      hypre_CSRMatrixI( device_diag ) == matrix.csr.row_ptr.device_pointer,
      "Device Hypre ParCSR row pointer does not alias GenDiL device storage." ) && success;
   success = Check(
      hypre_CSRMatrixJ( device_diag ) == matrix.csr.col_ind.device_pointer,
      "Device Hypre ParCSR column pointer does not alias GenDiL device storage." ) && success;
   success = Check(
      hypre_CSRMatrixData( device_diag ) == matrix.csr.values.device_pointer,
      "Device Hypre ParCSR value pointer does not alias GenDiL device storage." ) && success;

   Vector y_device( 3 );
   y_device = 0.0;
   Apply( HypreCSRDeviceBackend{}, matrix, x, y_device );
   success = Check(
      y_device.IsDeviceValid() && !y_device.IsHostValid(),
      "HypreCSRDeviceBackend apply should leave the output vector device-current." ) && success;
   success = CheckVector(
      y_device,
      { 11.0, 13.0, 13.0 },
      "HypreCSRDeviceBackend apply produced the wrong result." ) && success;
#endif

   return success;
}

bool TestHypreMatVecBackendMarkerDispatch()
{
   auto matrix = BuildSmallHypreCSRMatrix( HostCSRBackend<>{} );

   HYPRE_ParCSRMatrix host_parcsr =
      matrix.GetHostHypreParCSR();

   bool success = true;
   success = Check(
      matrix.GetHypreParCSR() == host_parcsr,
      "HostMatVecBackend marker dispatch should select the host Hypre shell." ) && success;
   success = Check(
      static_cast< HYPRE_ParCSRMatrix >( matrix ) == host_parcsr,
      "Hypre ParCSR conversion should use HostMatVecBackend marker dispatch." ) && success;

   auto * internal =
      reinterpret_cast< hypre_ParCSRMatrix * >( host_parcsr );
   hypre_CSRMatrix * diag =
      hypre_ParCSRMatrixDiag( internal );

   success = Check(
      hypre_CSRMatrixI( diag ) == matrix.csr.row_ptr.host_pointer,
      "HostMatVecBackend marker-dispatched shell should alias host row storage." ) && success;
   success = Check(
      hypre_CSRMatrixJ( diag ) == matrix.csr.col_ind.host_pointer,
      "HostMatVecBackend marker-dispatched shell should alias host column storage." ) && success;
   success = Check(
      hypre_CSRMatrixData( diag ) == matrix.csr.values.host_pointer,
      "HostMatVecBackend marker-dispatched shell should alias host value storage." ) && success;

   return success;
}

bool TestHypreMoveAfterAlias()
{
   auto matrix = BuildSmallHypreCSRMatrix();

   HYPRE_ParCSRMatrix original_shell =
      matrix.GetHostHypreParCSR();
   auto * original_internal =
      reinterpret_cast< hypre_ParCSRMatrix * >( original_shell );
   hypre_CSRMatrix * original_diag =
      hypre_ParCSRMatrixDiag( original_internal );

   const HYPRE_Int * original_row_ptr = hypre_CSRMatrixI( original_diag );
   const HYPRE_Int * original_col_ind = hypre_CSRMatrixJ( original_diag );
   const HYPRE_Complex * original_values =
      hypre_CSRMatrixData( original_diag );

#ifdef GENDIL_USE_HYPRE_DEVICE
   HYPRE_ParCSRMatrix original_device_shell =
      matrix.GetDeviceHypreParCSR();
#endif

   HypreCSRMatrix<> moved( std::move( matrix ) );

   bool success = true;
   success = Check(
      matrix.host_parcsr == nullptr,
      "Moved-from HypreCSRMatrix should no longer own the host Hypre shell." ) && success;
#ifdef GENDIL_USE_HYPRE_DEVICE
   success = Check(
      matrix.device_parcsr == nullptr,
      "Moved-from HypreCSRMatrix should no longer own the device Hypre shell." ) && success;
#endif

   HYPRE_ParCSRMatrix moved_shell =
      moved.GetHostHypreParCSR();
   success = Check(
      moved_shell == original_shell,
      "Move-after-alias should transfer the existing host Hypre shell." ) && success;

#ifdef GENDIL_USE_HYPRE_DEVICE
   HYPRE_ParCSRMatrix moved_device_shell =
      moved.GetDeviceHypreParCSR();
   success = Check(
      moved_device_shell == original_device_shell,
      "Move-after-alias should transfer the existing device Hypre shell." ) && success;
#endif

   auto * moved_internal =
      reinterpret_cast< hypre_ParCSRMatrix * >( moved_shell );
   hypre_CSRMatrix * moved_diag =
      hypre_ParCSRMatrixDiag( moved_internal );

   success = Check(
      hypre_CSRMatrixI( moved_diag ) == moved.csr.row_ptr.host_pointer,
      "Moved Hypre shell row pointer should alias destination CSR storage." ) && success;
   success = Check(
      hypre_CSRMatrixJ( moved_diag ) == moved.csr.col_ind.host_pointer,
      "Moved Hypre shell column pointer should alias destination CSR storage." ) && success;
   success = Check(
      hypre_CSRMatrixData( moved_diag ) == moved.csr.values.host_pointer,
      "Moved Hypre shell value pointer should alias destination CSR storage." ) && success;

   success = Check(
      moved.csr.row_ptr.host_pointer == original_row_ptr,
      "Destination CSR row pointer should retain the aliased storage address." ) && success;
   success = Check(
      moved.csr.col_ind.host_pointer == original_col_ind,
      "Destination CSR column pointer should retain the aliased storage address." ) && success;
   success = Check(
      moved.csr.values.host_pointer == original_values,
      "Destination CSR value pointer should retain the aliased storage address." ) && success;

   Vector x( 3 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 1.0;
   x_data[1] = 2.0;
   x_data[2] = 3.0;

   Vector y_hypre( 3 );
   y_hypre = 0.0;
   HypreMatvec( moved, x, y_hypre );

   success = CheckVector(
      y_hypre,
      { 11.0, 13.0, 13.0 },
      "HypreMatvec after move-after-alias produced the wrong result." ) && success;

   return success;
}

} // namespace

int main( int argc, char ** argv )
{
   hypre_MPI_Init( &argc, &argv );

   bool success = true;
   {
      HypreSession hypre;
      success = TestHypreCSRFinalizationAndNativeApply() && success;
      success = TestHypreCSRDiagonalFirstSortReduce() && success;
      success = TestHypreAliasVectorViewsAndMatvec() && success;
      success = TestHypreMatVecBackendMarkerDispatch() && success;
      success = TestHypreMoveAfterAlias() && success;
   }

   hypre_MPI_Finalize();
   return success ? 0 : 1;
}
