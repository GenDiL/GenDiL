// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>
#include <ostream>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MathHelperFunctions/sqrt.hpp"

namespace gendil {

/**
 * @brief A structure describing a point in an arbitrary dimension space.
 * 
 * @tparam Dim The dimension of the space embedding the point.
 */
template < size_t Dim >
struct Point
{
   static constexpr size_t space_dim = Dim;
   static constexpr size_t geometry_dim = 0;
   static constexpr size_t num_nodes = 1;
   static constexpr size_t num_faces = 0;

   Real coordinates [ Dim ];

   constexpr Point() : coordinates{} { }

   /// Copy a full coordinate array, including implicit conversion in geometry APIs.
   // Deduce N so brace-only coordinate lists keep their existing overloads.
   template < size_t N >
      requires ( N == Dim )
   constexpr Point( const std::array< Real, N > & values ) : coordinates{}
   {
      for ( size_t i = 0; i < Dim; ++i )
      {
         coordinates[i] = values[i];
      }
   }

   // Automatically cast a Real into a one-dimensional Point.
   constexpr Point( Real arg ) requires ( Dim == 1 ) : coordinates{ arg }
   { }

   template < typename... Args >
      requires ( Dim != 1 )
   constexpr Point( Args... args ) : coordinates{ args... }
   {
      static_assert(
         sizeof...(Args) == Dim,
         "Wrong number of coordinates"
      );
   }

   // Automatically cast a one-dimensional Point into a Real.
   constexpr operator Real() const requires ( Dim == 1 )
   {
      return coordinates[ 0 ];
   }

   constexpr Real operator[]( size_t index ) const
   {
      // TODO: Add debug assertion?
      return coordinates[ index ];
   }

   constexpr Real& operator[]( size_t index )
   {
      // TODO: Add debug assertion?
      return coordinates[ index ];
   }
};

template < size_t Dim >
std::ostream& operator<<(std::ostream& os, Point<Dim> const & index) {
   os << "( ";
   for (size_t i = 0; i < Dim; i++)
   {
      os << index[ i ] << " ";
   }
   os << ")";
   return os;
}

template < size_t Dim >
bool operator==( Point<Dim> const & a, Point<Dim> const & b )
{
   bool equal = true;
   for ( size_t i = 0; i < Dim; ++i )
   {
      equal = equal && (a[i] == b[i]);
   }
   return equal;
}

template < size_t Dim >
auto operator+( Point<Dim> const & a, Point<Dim> const & b )
{
   Point<Dim> result;
   for ( size_t i = 0; i < Dim; ++i )
   {
      result[i] = a[i] + b[i];
   }
   return result;
}

template < size_t Dim >
auto operator-( Point<Dim> const & a, Point<Dim> const & b )
{
   Point<Dim> result;
   for ( size_t i = 0; i < Dim; ++i )
   {
      result[i] = a[i] - b[i];
   }
   return result;
}

template < size_t Dim >
auto operator*( std::array<Real,Dim> const & a, Point<Dim> const & b )
{
   Point<Dim> result;
   for ( size_t i = 0; i < Dim; ++i )
   {
      result[i] = a[i] * b[i];
   }
   return result;
}

template < size_t Dim >
Real Distance( Point<Dim> const & a, Point<Dim> const & b )
{
   Real result = 0.0;
   for ( size_t i = 0; i < Dim; ++i )
   {
      Real d = a[i] - b[i];
      result += d * d;
   }
   return Sqrt(result);
}

template < size_t Dim >
bool AlmostEqual( Point<Dim> const & a, Point<Dim> const & b )
{
   const Real tol = 1e-8;
   return Distance(a, b) < tol;
}

}
