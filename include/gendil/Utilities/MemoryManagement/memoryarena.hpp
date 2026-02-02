// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil
{

template < typename T, size_t Size >
struct MemoryArena
{
   T * memory;
   mutable size_t offset;

   GENDIL_HOST_DEVICE
   MemoryArena( T * mem ) : memory{mem}, offset{0}
   {
   }

   GENDIL_HOST_DEVICE
   T * allocate( size_t size ) const
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
   void reset() const
   {
      offset = 0;
   }
};

} // namespace gendil
