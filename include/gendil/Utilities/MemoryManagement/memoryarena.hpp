// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstdio>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/debug.hpp"

#if defined( GENDIL_USE_CUDA )
#define GENDIL_MEMORY_ARENA_NOINLINE __noinline__
#elif defined( __GNUC__ ) || defined( __clang__ )
#define GENDIL_MEMORY_ARENA_NOINLINE __attribute__((noinline))
#else
#define GENDIL_MEMORY_ARENA_NOINLINE
#endif

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

      // Keep this hot path shaped like the original runtime allocator. HIP
      // codegen has been sensitive to heavier overflow handling here.
      if ( offset + size <= Size )
      {
         ptr = memory + offset;
         offset += size;
      }
      else
      {
         MemoryArenaOverflow( size, offset, Size );
      }

      return ptr;
   }

   static GENDIL_HOST_DEVICE GENDIL_MEMORY_ARENA_NOINLINE
   void MemoryArenaOverflow(
      size_t requested,
      size_t current_offset,
      size_t capacity )
   {
      const size_t remaining =
         ( current_offset < capacity ) ? ( capacity - current_offset ) : 0;

      printf(
         "GenDiL MemoryArena overflow: requested=%llu, offset=%llu, "
         "remaining=%llu, capacity=%llu\n",
         static_cast<unsigned long long>(requested),
         static_cast<unsigned long long>(current_offset),
         static_cast<unsigned long long>(remaining),
         static_cast<unsigned long long>(capacity)
      );

#if defined( __CUDA_ARCH__ )
      __trap();
#elif defined( __HIP_DEVICE_COMPILE__ )
      __builtin_trap();
#else
      std::abort();
#endif
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

#undef GENDIL_MEMORY_ARENA_NOINLINE
