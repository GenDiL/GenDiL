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
      << "test-csr-device-apply skipped because GENDIL_USE_DEVICE is not enabled.\n";
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

auto MakeDeviceCSRMatrix()
{
   auto matrix =
      MakeCSRMatrix< Real, GlobalIndex >(
         4,
         5,
         6,
         NativeDeviceCSRBackend{} );

   matrix.row_ptr[0] = 0;
   matrix.row_ptr[1] = 2;
   matrix.row_ptr[2] = 5;
   matrix.row_ptr[3] = 5;
   matrix.row_ptr[4] = 6;

   matrix.col_ind[0] = 0;
   matrix.values[0] = 2.0;

   matrix.col_ind[1] = 3;
   matrix.values[1] = -1.0;

   matrix.col_ind[2] = 1;
   matrix.values[2] = 4.0;

   matrix.col_ind[3] = 2;
   matrix.values[3] = 0.0;

   matrix.col_ind[4] = 4;
   matrix.values[4] = 1.5;

   matrix.col_ind[5] = 0;
   matrix.values[5] = -3.0;

   ToDevice( matrix.num_rows + 1, matrix.row_ptr );
   ToDevice( matrix.nnz, matrix.col_ind );
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

bool TestNativeDeviceCSRAction()
{
   auto matrix = MakeDeviceCSRMatrix();

   Vector x( 5 );
   FillInput( x );

   Vector y_host( 4 );
   Vector y_device_apply( 4 );
   Vector y_device_operator( 4 );

   y_host = -77.0;
   y_device_apply = -77.0;
   y_device_operator = -77.0;

   Apply( HostCSRBackend{}, matrix, x, y_host );
   Apply( NativeDeviceCSRBackend{}, matrix, x, y_device_apply );
   matrix( x, y_device_operator );

   bool success = true;
   success = CheckVectorsNear(
      y_device_apply,
      y_host,
      "NativeDeviceCSRBackend Apply disagrees with HostCSRBackend Apply." ) && success;
   success = CheckVectorsNear(
      y_device_operator,
      y_host,
      "CSRMatrix::operator()(x, y) with NativeDeviceCSRBackend disagrees with host Apply." ) && success;

   FreeCSRMatrix( matrix );
   return success;
}

} // namespace

int main()
{
   return TestNativeDeviceCSRAction() ? 0 : 1;
}

#endif
