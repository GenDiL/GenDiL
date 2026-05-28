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
      if ( offset + size <= Size )
      {
         T * ptr = memory + offset;
         offset += size;
         return ptr;
      }

      const size_t remaining = ( offset < Size ) ? ( Size - offset ) : 0;

      printf(
         "GenDiL MemoryArena overflow: requested=%llu, offset=%llu, "
         "remaining=%llu, capacity=%llu\n",
         static_cast<unsigned long long>(size),
         static_cast<unsigned long long>(offset),
         static_cast<unsigned long long>(remaining),
         static_cast<unsigned long long>(Size)
      );

#if defined( __CUDA_ARCH__ )
      __trap();
#elif defined( __HIP_DEVICE_COMPILE__ )
      __builtin_trap();
#else
      std::abort();
#endif

      return nullptr;
   }

   template < size_t RequestSize >
   GENDIL_HOST_DEVICE
   T * allocate() const
   {
      static_assert(
         RequestSize <= Size,
         "GenDiL shared-memory arena is too small for this compile-time "
         "allocation." );

      return allocate( RequestSize );
   }

   GENDIL_HOST_DEVICE
   void reset() const
   {
      offset = 0;
   }
};

} // namespace gendil
