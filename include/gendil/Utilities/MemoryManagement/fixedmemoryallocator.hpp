// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil
{

/**
 * @brief An allocator where the "heap" is an array of some fixed size. For
 * simplicity the allocator only allocates single blocks of size BlockSize. The
 * array may be any array type (e.g. static array, dynamic array, shared memory
 * array) as long as it holds BlockSize * NumBlocks units of T and remains
 * allocated for the lifetime of the FixedMemoryAllocator.
*/
template < typename T, size_t BlockSize, size_t NumBlocks >
class FixedMemoryAllocator
{
public:
   GENDIL_HOST_DEVICE
   FixedMemoryAllocator( T * mem ) : memory{mem}
   {
      ConstexprLoop< NumBlocks >( [&] ( auto i ) { occupied[ i ] = false; });
   }

   GENDIL_HOST_DEVICE
   T * allocate() const
   {
      T * ptr = nullptr;

      for ( size_t i = 0; i < NumBlocks; ++i )
      {
         if ( not occupied[ i ] )
         {
            occupied[ i ] = true;
            ptr = memory + i * BlockSize;
            break;
         }
      }

      Assert(ptr != nullptr, "out of blocks in fixed memory!");
      return ptr;
   }

   GENDIL_HOST_DEVICE
   void deallocate( T * ptr ) const
   {
      const Integer i = (ptr - memory) / BlockSize;

      Assert( i < NumBlocks, "provided pointer is not managed by this allocator!");
      occupied[ i ] = false;
   }

private:
   mutable bool occupied[ NumBlocks ];
   T * const memory;
};

} // namespace gendil
