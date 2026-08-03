// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

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
   return std::abs( lhs - rhs ) < 1.0e-12;
}

void FillInputAndOutput( Vector & x, Vector & y )
{
   Real * x_data = x.WriteHostData();
   x_data[0] = 4.0;
   x_data[1] = 5.0;

   Real * y_data = y.WriteHostData();
   y_data[0] = 10.0;
   y_data[1] = 20.0;
}

bool CheckAddResult( const Vector & y, const char * message )
{
   const Real * data = y.ReadHostData();
   return
      Check( Near( data[0], 18.0 ), message ) &&
      Check( Near( data[1], 35.0 ), message );
}

template < typename Matrix, typename Backend >
bool CheckApplyAdd(
   const Backend & backend,
   const Matrix & matrix,
   const char * explicit_message,
   const char * stored_message )
{
   Vector x( 2 );
   Vector explicit_y( 2 );
   FillInputAndOutput( x, explicit_y );
   ApplyAdd( backend, matrix, x, explicit_y );

   Vector stored_y( 2 );
   FillInputAndOutput( x, stored_y );
   ApplyAdd( matrix, x, stored_y );

   return
      CheckAddResult( explicit_y, explicit_message ) &&
      CheckAddResult( stored_y, stored_message );
}

#ifdef GENDIL_USE_MFEM
template < typename InputVector, typename OutputVector, typename Matrix, typename Backend >
bool CheckMixedVectorApply(
   const Backend & backend,
   const Matrix & matrix,
   const char * message )
{
   InputVector x( 2 );
   auto * x_data = WriteHostVector( x );
   x_data[0] = 4.0;
   x_data[1] = 5.0;

   auto check_overwrite = [&] ( const OutputVector & y )
   {
      const auto * data = ReadHostVector( y );
      return Near( data[0], 8.0 ) && Near( data[1], 15.0 );
   };
   auto initialize_add = [] ( OutputVector & y )
   {
      auto * data = WriteHostVector( y );
      data[0] = 10.0;
      data[1] = 20.0;
   };
   auto check_add = [&] ( const OutputVector & y )
   {
      const auto * data = ReadHostVector( y );
      return Near( data[0], 18.0 ) && Near( data[1], 35.0 );
   };

   OutputVector explicit_y( 2 );
   Apply( backend, matrix, x, explicit_y );
   bool success = Check( check_overwrite( explicit_y ), message );

   OutputVector stored_y( 2 );
   matrix( x, stored_y );
   success = Check( check_overwrite( stored_y ), message ) && success;

   initialize_add( explicit_y );
   ApplyAdd( backend, matrix, x, explicit_y );
   success = Check( check_add( explicit_y ), message ) && success;

   initialize_add( stored_y );
   ApplyAdd( matrix, x, stored_y );
   success = Check( check_add( stored_y ), message ) && success;
   return success;
}

template < typename Matrix, typename Backend >
bool CheckMFEMApplyAdd(
   const Backend & backend,
   const Matrix & matrix,
   const char * explicit_message,
   const char * stored_message )
{
   mfem::Vector x( 2 );
   auto * x_data = x.HostWrite();
   x_data[0] = 4.0;
   x_data[1] = 5.0;

   mfem::Vector explicit_y( 2 );
   auto * explicit_data = explicit_y.HostWrite();
   explicit_data[0] = 10.0;
   explicit_data[1] = 20.0;
   ApplyAdd( backend, matrix, x, explicit_y );

   mfem::Vector stored_y( 2 );
   auto * stored_data = stored_y.HostWrite();
   stored_data[0] = 10.0;
   stored_data[1] = 20.0;
   ApplyAdd( matrix, x, stored_y );

   explicit_data = explicit_y.HostReadWrite();
   stored_data = stored_y.HostReadWrite();
   return
      Check(
         Near( explicit_data[0], 18.0 ) &&
         Near( explicit_data[1], 35.0 ),
         explicit_message ) &&
      Check(
         Near( stored_data[0], 18.0 ) &&
         Near( stored_data[1], 35.0 ),
         stored_message );
}
#endif

bool TestBSR()
{
   auto matrix =
      MakeBlockDiagonalDGBSRPattern<
         Real,
         GlobalIndex,
         BlockLayout::RowMajor,
         HostBSRBackend<> >( 2, 1, 1 );
   auto values = GetHostValuesWriteView( matrix );
   values.values[0] = 2.0;
   values.values[1] = 3.0;

   bool success = CheckApplyAdd(
      matrix.backend,
      matrix,
      "Explicit host BSR ApplyAdd produced the wrong result.",
      "Stored host BSR ApplyAdd produced the wrong result." );
#ifdef GENDIL_USE_MFEM
   success = CheckMFEMApplyAdd(
      matrix.backend,
      matrix,
      "Explicit host BSR MFEM ApplyAdd produced the wrong result.",
      "Stored host BSR MFEM ApplyAdd produced the wrong result." ) && success;
   success = CheckMixedVectorApply< Vector, mfem::Vector >(
      matrix.backend,
      matrix,
      "BSR GenDiL-input/MFEM-output apply produced the wrong result." ) &&
      success;
   success = CheckMixedVectorApply< mfem::Vector, Vector >(
      matrix.backend,
      matrix,
      "BSR MFEM-input/GenDiL-output apply produced the wrong result." ) &&
      success;
#endif
   return success;
}

bool TestCOO()
{
   auto matrix =
      MakeCOOMatrix<
         Real,
         GlobalIndex,
         HostCOOBackend<> >( 2, 2, 2 );
   auto view = GetHostWriteView( matrix );
   view.rows[0] = 0;
   view.cols[0] = 0;
   view.values[0] = 2.0;
   view.rows[1] = 1;
   view.cols[1] = 1;
   view.values[1] = 3.0;

   bool success = CheckApplyAdd(
      matrix.backend,
      matrix,
      "Explicit host COO ApplyAdd produced the wrong result.",
      "Stored host COO ApplyAdd produced the wrong result." );
#ifdef GENDIL_USE_MFEM
   success = CheckMFEMApplyAdd(
      matrix.backend,
      matrix,
      "Explicit host COO MFEM ApplyAdd produced the wrong result.",
      "Stored host COO MFEM ApplyAdd produced the wrong result." ) && success;
   success = CheckMixedVectorApply< Vector, mfem::Vector >(
      matrix.backend,
      matrix,
      "COO GenDiL-input/MFEM-output apply produced the wrong result." ) &&
      success;
   success = CheckMixedVectorApply< mfem::Vector, Vector >(
      matrix.backend,
      matrix,
      "COO MFEM-input/GenDiL-output apply produced the wrong result." ) &&
      success;
#endif
   return success;
}

bool TestCSR()
{
   auto matrix =
      MakeCSRMatrix<
         Real,
         GlobalIndex,
         HostCSRBackend<> >( 2, 2, 2 );
   auto view = GetHostWriteView( matrix );
   view.row_ptr[0] = 0;
   view.row_ptr[1] = 1;
   view.row_ptr[2] = 2;
   view.col_ind[0] = 0;
   view.col_ind[1] = 1;
   view.values[0] = 2.0;
   view.values[1] = 3.0;

   bool success = CheckApplyAdd(
      matrix.backend,
      matrix,
      "Explicit host CSR ApplyAdd produced the wrong result.",
      "Stored host CSR ApplyAdd produced the wrong result." );
#ifdef GENDIL_USE_MFEM
   success = CheckMFEMApplyAdd(
      matrix.backend,
      matrix,
      "Explicit host CSR MFEM ApplyAdd produced the wrong result.",
      "Stored host CSR MFEM ApplyAdd produced the wrong result." ) && success;
   success = CheckMixedVectorApply< Vector, mfem::Vector >(
      matrix.backend,
      matrix,
      "CSR GenDiL-input/MFEM-output apply produced the wrong result." ) &&
      success;
   success = CheckMixedVectorApply< mfem::Vector, Vector >(
      matrix.backend,
      matrix,
      "CSR MFEM-input/GenDiL-output apply produced the wrong result." ) &&
      success;
#endif
   return success;
}

bool TestCSC()
{
   auto matrix =
      MakeCSCMatrix<
         Real,
         GlobalIndex,
         HostCSCBackend<> >( 2, 2, 2 );
   auto view = GetHostWriteView( matrix );
   view.col_ptr[0] = 0;
   view.col_ptr[1] = 1;
   view.col_ptr[2] = 2;
   view.row_ind[0] = 0;
   view.row_ind[1] = 1;
   view.values[0] = 2.0;
   view.values[1] = 3.0;

   bool success = CheckApplyAdd(
      matrix.backend,
      matrix,
      "Explicit host CSC ApplyAdd produced the wrong result.",
      "Stored host CSC ApplyAdd produced the wrong result." );
#ifdef GENDIL_USE_MFEM
   success = CheckMFEMApplyAdd(
      matrix.backend,
      matrix,
      "Explicit host CSC MFEM ApplyAdd produced the wrong result.",
      "Stored host CSC MFEM ApplyAdd produced the wrong result." ) && success;
   success = CheckMixedVectorApply< Vector, mfem::Vector >(
      matrix.backend,
      matrix,
      "CSC GenDiL-input/MFEM-output apply produced the wrong result." ) &&
      success;
   success = CheckMixedVectorApply< mfem::Vector, Vector >(
      matrix.backend,
      matrix,
      "CSC MFEM-input/GenDiL-output apply produced the wrong result." ) &&
      success;
#endif
   return success;
}

bool TestEmptyCOO()
{
   auto matrix =
      MakeCOOMatrix<
         Real,
         GlobalIndex,
         HostCOOBackend<> >( 2, 2, 0 );
   Vector x( 2 );
   Vector y( 2 );
   FillInputAndOutput( x, y );

   const bool host_valid_before = y.IsHostValid();
   const bool device_valid_before = y.IsDeviceValid();
   ApplyAdd( matrix, x, y );
   bool success = Check(
      Near( y.ReadHostData()[0], 10.0 ) &&
      Near( y.ReadHostData()[1], 20.0 ),
      "Empty COO ApplyAdd changed the output." );
   success = Check(
      host_valid_before && !device_valid_before,
      "Empty COO validity fixture did not begin host-current." ) && success;
   Vector validity_y( 2 );
   FillInputAndOutput( x, validity_y );
   const bool expected_host_valid = validity_y.IsHostValid();
   const bool expected_device_valid = validity_y.IsDeviceValid();
   ApplyAdd( matrix, x, validity_y );
   success = Check(
      validity_y.IsHostValid() == expected_host_valid &&
      validity_y.IsDeviceValid() == expected_device_valid,
      "Empty COO ApplyAdd changed output validity state." ) && success;

   Vector overwrite( 2 );
   Apply( matrix.backend, matrix, x, overwrite );
   success = Check(
      Near( overwrite.ReadHostData()[0], 0.0 ) &&
      Near( overwrite.ReadHostData()[1], 0.0 ),
      "Empty COO Apply did not initialize the output to zero." ) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestBSR() && success;
   success = TestCOO() && success;
   success = TestCSR() && success;
   success = TestCSC() && success;
   success = TestEmptyCOO() && success;
   return success ? 0 : 1;
}
