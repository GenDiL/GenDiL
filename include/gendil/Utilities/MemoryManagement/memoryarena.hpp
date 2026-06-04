// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstdio>
#include <type_traits>

#include "gendil/Utilities/types.hpp"

namespace gendil
{

template < typename T, size_t Size >
struct MemoryArena
{
   T * memory;
   size_t offset;

   GENDIL_HOST_DEVICE
   MemoryArena( T * mem ) : memory{mem}, offset{0}
   {
   }

   GENDIL_HOST_DEVICE
   T * allocate( size_t size )
   {
      T * ptr = nullptr;

      if ( offset + size <= Size )
      {
         ptr = memory + offset;
         offset += size;
      }
      else
      {
         // Out of memory, return nullptr
         ptr = nullptr;
         printf(
            "Memory arena is full! Requested size: %llu, available size: %llu\n",
            static_cast<unsigned long long>(size),
            static_cast<unsigned long long>(Size - offset)
         );
      }

      return ptr;
   }

   GENDIL_HOST_DEVICE
   size_t mark() const
   {
      return offset;
   }

   GENDIL_HOST_DEVICE
   void restore( size_t mark )
   {
      offset = mark;
   }

   GENDIL_HOST_DEVICE
   void reset()
   {
      offset = 0;
   }
};

template < typename Arena >
struct MemoryArenaCapacity;

template < typename T, size_t Size >
struct MemoryArenaCapacity< MemoryArena< T, Size > >
{
   static constexpr size_t value = Size;
};

} // namespace gendil

#define GENDIL_CHECK_MEMORY_ARENA_REQUEST(ALLOCATOR, REQUEST_SIZE) \
   static_assert( \
      ( REQUEST_SIZE ) <= \
         ::gendil::MemoryArenaCapacity< \
            std::remove_cvref_t< decltype( ALLOCATOR ) > >::value, \
      "GenDiL shared-memory arena is too small for this compile-time " \
      "allocation." )
