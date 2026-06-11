// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-csc-device-apply skipped because GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

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

auto MakeDeviceCSCMatrix()
{
   auto matrix =
      MakeCSCMatrix< Real, GlobalIndex >(
         4,
         6,
         6,
         NativeDeviceCSCBackend{} );

   matrix.col_ptr[0] = 0;
   matrix.col_ptr[1] = 2;
   matrix.col_ptr[2] = 3;
   matrix.col_ptr[3] = 4;
   matrix.col_ptr[4] = 5;
   matrix.col_ptr[5] = 6;
   matrix.col_ptr[6] = 6;

   matrix.row_ind[0] = 0;
   matrix.values[0] = 2.0;

   matrix.row_ind[1] = 3;
   matrix.values[1] = -3.0;

   matrix.row_ind[2] = 1;
   matrix.values[2] = 4.0;

   matrix.row_ind[3] = 1;
   matrix.values[3] = 0.0;

   matrix.row_ind[4] = 0;
   matrix.values[4] = -1.0;

   matrix.row_ind[5] = 1;
   matrix.values[5] = 1.5;

   ToDevice( matrix.num_cols + 1, matrix.col_ptr );
   ToDevice( matrix.nnz, matrix.row_ind );
   ToDevice( matrix.nnz, matrix.values );

   return matrix;
}

void FillInput( Vector & x )
{
   Real * x_data = x.WriteHostData();
   x_data[0] = 1.0;
   x_data[1] = 2.0;
   x_data[2] = 3.0;
   x_data[3] = 4.0;
   x_data[4] = 5.0;
   x_data[5] = 6.0;
}

bool CheckVectorsNear(
   const Vector & actual,
   const Vector & expected,
   const char * message )
{
   const Real * actual_data = actual.ReadHostData();
   const Real * expected_data = expected.ReadHostData();

   bool success = true;
   for ( GlobalIndex i = 0; i < static_cast< GlobalIndex >( actual.Size() ); ++i )
   {
      success = Check(
         Near( actual_data[i], expected_data[i] ),
         message ) && success;
   }
   return success;
}

bool TestNativeDeviceCSCAction()
{
   auto matrix = MakeDeviceCSCMatrix();

   Vector x( 6 );
   FillInput( x );

   Vector y_host( 4 );
   Vector y_device_apply( 4 );
   Vector y_device_operator( 4 );

   y_host = -77.0;
   y_device_apply = -77.0;
   y_device_operator = -77.0;

   Apply( HostCSCBackend{}, matrix, x, y_host );
   Apply( NativeDeviceCSCBackend{}, matrix, x, y_device_apply );
   matrix( x, y_device_operator );

   bool success = true;
   success = CheckVectorsNear(
      y_device_apply,
      y_host,
      "NativeDeviceCSCBackend Apply disagrees with HostCSCBackend Apply." ) && success;
   success = CheckVectorsNear(
      y_device_operator,
      y_host,
      "CSCMatrix::operator()(x, y) with NativeDeviceCSCBackend disagrees with host Apply." ) && success;

   FreeCSCMatrix( matrix );
   return success;
}

} // namespace

int main()
{
   return TestNativeDeviceCSCAction() ? 0 : 1;
}

#endif
