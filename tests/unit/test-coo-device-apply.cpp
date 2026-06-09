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
      << "test-coo-device-apply skipped because GENDIL_USE_DEVICE is not enabled.\n";
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

auto MakeDeviceCOOMatrix()
{
   auto matrix =
      MakeCOOMatrix< Real, GlobalIndex >(
         4,
         3,
         6,
         NativeDeviceCOOBackend{} );

   matrix.rows[0] = 0;
   matrix.cols[0] = 0;
   matrix.values[0] = 1.0;

   matrix.rows[1] = 0;
   matrix.cols[1] = 2;
   matrix.values[1] = 4.0;

   matrix.rows[2] = 1;
   matrix.cols[2] = 0;
   matrix.values[2] = 1.5;

   matrix.rows[3] = 1;
   matrix.cols[3] = 1;
   matrix.values[3] = -1.0;

   matrix.rows[4] = 1;
   matrix.cols[4] = 2;
   matrix.values[4] = 0.0;

   matrix.rows[5] = 3;
   matrix.cols[5] = 1;
   matrix.values[5] = -2.0;

   ToDevice( matrix.nnz, matrix.rows );
   ToDevice( matrix.nnz, matrix.cols );
   ToDevice( matrix.nnz, matrix.values );

   return matrix;
}

void FillInput( Vector & x )
{
   Real * x_data = x.WriteHostData();
   x_data[0] = 2.0;
   x_data[1] = 3.0;
   x_data[2] = 5.0;
}

bool CheckVector(
   const Vector & vector,
   const Real expected0,
   const Real expected1,
   const Real expected2,
   const Real expected3,
   const char * message )
{
   const Real expected[] = {
      expected0,
      expected1,
      expected2,
      expected3
   };

   const Real * data = vector.ReadHostData();
   bool success = true;
   for ( GlobalIndex i = 0; i < 4; ++i )
   {
      success = Check( Near( data[i], expected[i] ), message ) && success;
   }
   return success;
}

bool CheckVectorsNear(
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

bool TestNativeDeviceCOOAction()
{
   auto matrix = MakeDeviceCOOMatrix();

   Vector x( 3 );
   FillInput( x );

   Vector y_host( 4 );
   Vector y_device_apply( 4 );
   Vector y_device_operator( 4 );

   y_host = -77.0;
   y_device_apply = -77.0;
   y_device_operator = -77.0;

   Apply( HostCOOBackend{}, matrix, x, y_host );
   Apply( NativeDeviceCOOBackend{}, matrix, x, y_device_apply );
   matrix( x, y_device_operator );

   bool success = true;
   success = CheckVector(
      y_device_apply,
      22.0,
      0.0,
      0.0,
      -6.0,
      "NativeDeviceCOOBackend Apply produced the wrong result." ) && success;
   success = CheckVectorsNear(
      y_device_apply,
      y_host,
      "NativeDeviceCOOBackend Apply disagrees with HostCOOBackend Apply." ) && success;
   success = CheckVectorsNear(
      y_device_operator,
      y_host,
      "COOMatrix::operator()(x, y) with NativeDeviceCOOBackend disagrees with host Apply." ) && success;

   FreeCOOMatrix( matrix );
   return success;
}

} // namespace

int main()
{
   return TestNativeDeviceCOOAction() ? 0 : 1;
}

#endif
