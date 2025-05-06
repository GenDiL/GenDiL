// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil
{

namespace details
{
   // Always perform "empty" loop
   template < typename Lambda, typename... Args >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   void UnitLoopImpl( Lambda && body, std::index_sequence<>, Args... args )
   {
      body( std::forward<Args>(args)... );
   }

   template < typename Lambda, size_t HeadBound, size_t... TailBounds, typename... Args >
   GENDIL_HOST_DEVICE
   void UnitLoopImpl( Lambda && body, std::index_sequence<HeadBound, TailBounds...>, Args... args )
   {
      for ( GlobalIndex index = 0; index < HeadBound; ++index )
      {
         UnitLoopImpl( std::forward<Lambda>(body), std::index_sequence<TailBounds...>{}, index, std::forward<Args>(args)... );
      }
   }
} // namespace details

/**
 * @brief Nested for loop abstraction with first index running fastest. If the
 * bounds are empty, then the unit loop calls the body once (with no index).
*/
template < size_t... Bounds, typename Lambda, typename... Args >
GENDIL_HOST_DEVICE GENDIL_INLINE
void UnitLoop( Lambda && body, Args... args )
{
   using bnds = ReverseIndexSequence< Bounds... >;
   details::UnitLoopImpl( std::forward<Lambda>(body), bnds{}, std::forward<Args>(args)... );
}

/**
 * @brief Nested for loop abstraction with first index running fastest. If the
 * bounds are empty, then the unit loop calls the body once (with no index).
 * Here the bounds can be supplied as an std::index_sequence type.
*/
template < typename BoundsSequence, typename Lambda, typename... Args>
GENDIL_HOST_DEVICE GENDIL_INLINE
void UnitLoop( Lambda && body, Args... args )
{
   using bnds = typename ReverseIndexSequence_t< BoundsSequence >::type;
   details::UnitLoopImpl( std::forward<Lambda>(body), bnds{}, std::forward<Args>(args)... );
}

} // namespace gendil
