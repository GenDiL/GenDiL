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

      Assert(ptr != nullptr, "out of memory in memory arena!");
      return ptr;
   }

   GENDIL_HOST_DEVICE
   void reset() const
   {
      offset = 0;
   }
};

} // namespace gendil
