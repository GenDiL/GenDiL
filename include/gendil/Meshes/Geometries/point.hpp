// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <ostream>
#include "gendil/Utilities/types.hpp"

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

   template < typename... Args >
   constexpr Point( Args... args ) : coordinates{ args... }
   {
      static_assert(
         sizeof...(Args) == Dim,
         "Wrong number of coordinates"
      );
   }

   constexpr Real operator[]( size_t index ) const
   {
      // TODO: Add debug assertion?
      return coordinates[ index ];
   }
};

// Specialize the scalar case
template <>
struct Point<1>
{
   static constexpr size_t space_dim = 1;
   static constexpr size_t geometry_dim = 0;
   static constexpr size_t num_nodes = 1;
   static constexpr size_t num_faces = 0;

   Real coordinates [ space_dim ];

   constexpr Point() : coordinates{} { }

   // Automatically cast a Real into a Point<1>.
   constexpr Point( Real arg ) : coordinates{ arg }
   { }

   // Automatically cast a Point<1> into a Real.
   constexpr operator Real() const
   {
      return coordinates[ 0 ];
   }

   constexpr Real operator[]( size_t index ) const
   {
      // TODO: Add debug assertion?
      return coordinates[ 0 ];
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

// TODO: defines simple operations on Point, e.g. +,-?

}
