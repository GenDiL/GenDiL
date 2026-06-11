// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

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

void FillInput( Vector & x )
{
   Real * x_data = x.WriteHostData();
   x_data[0] = 2.0;
   x_data[1] = 3.0;
   x_data[2] = 5.0;
}

void FillSentinel( Vector & y )
{
   Real * y_data = y.WriteHostData();
   for ( GlobalIndex i = 0; i < y.Size(); ++i )
   {
      y_data[i] = -100.0 - static_cast< Real >( i );
   }
}

bool TestHandWrittenCOOAction()
{
   auto matrix = MakeCOOMatrix< Real, GlobalIndex >( 4, 3, 6 );

   // Canonical COO: sorted, unique coordinates. Row 2 is intentionally empty,
   // and entry (1, 2) is an explicit zero retained by the matrix.
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

   Vector x( 3 );
   FillInput( x );

   Vector y_apply( 4 );
   FillSentinel( y_apply );
   Apply( HostCOOBackend{}, matrix, x, y_apply );

   bool success = true;
   success = CheckVector(
      y_apply,
      22.0,
      0.0,
      0.0,
      -6.0,
      "Apply(HostCOOBackend, COOMatrix, x, y) produced the wrong result." ) && success;

   Vector y_operator( 4 );
   FillSentinel( y_operator );
   matrix( x, y_operator );

   success = CheckVector(
      y_operator,
      22.0,
      0.0,
      0.0,
      -6.0,
      "COOMatrix::operator()(x, y) produced the wrong result." ) && success;

   FreeCOOMatrix( matrix );
   return success;
}

} // namespace

int main()
{
   return TestHandWrittenCOOAction() ? 0 : 1;
}
