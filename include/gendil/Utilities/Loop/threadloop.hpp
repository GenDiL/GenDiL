// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

namespace gendil
{
   namespace details
   {
      // TODO Do a ThreadLoop( RegisterLoop ) Pattern?
      template <size_t I, typename ThreadLayout, typename Lambda, typename... Args >
      GENDIL_HOST_DEVICE GENDIL_INLINE
      void ReverseThreadLoopImpl( const ThreadLayout & thread, Lambda && body, std::index_sequence<>, Args... args )
      {
         if constexpr ( I >= ThreadLayout::thread_block_dim ) // TODO ThreadLayout::space_dim
         {
            body( std::forward< Args >( args )... );
         }
         else // Handle the case where we have more threaded dimensions than loop bounds
         {
            if ( thread.template GetThreadIndex< I >() == 0 ) // TODO is_threaded_v< I, KernelConfiguration >
            {
               ReverseThreadLoopImpl< I+1 >( thread, std::forward< Lambda >( body ), std::index_sequence<>{}, std::forward< Args >( args )... );
            }
         }
      }

      template <size_t I, typename ThreadLayout, typename Lambda, size_t Bound, size_t... Bounds, typename... Args >
      GENDIL_HOST_DEVICE GENDIL_INLINE
      void ReverseThreadLoopImpl( const ThreadLayout & thread, Lambda && body, std::index_sequence<Bound, Bounds...>, Args && ... args )
      {
         GlobalIndex begin, delta;

         // !FIXME: Assumes the threaded dimensions are the first dimensions
         if constexpr ( I < ThreadLayout::thread_block_dim ) // TODO is_threaded_v< I, KernelConfiguration >
         {
            begin = thread.template GetThreadIndex< I >();
            delta = ThreadLayout::template GetBlockDim< I >();
         }
         else
         {
            begin = 0;
            delta = 1;
         }

         for ( GlobalIndex index = begin; index < Bound; index += delta )
         {
            ReverseThreadLoopImpl< I+1 >( thread, std::forward< Lambda >( body ), std::index_sequence< Bounds... >{}, std::forward< Args >( args )..., index );
         }
      }

   } // namespace details

/**
 * @brief Arbitrary dimensional loop with last index running fastest. The first
 * ThreadLayout::Dim loop dimensions are parallelized by the threads.
 * 
 * @details example
 * ```
 * constexpr Integer Bx = 10, By = 10;
 * ThreadLayout< Bx, By > thread;
 * 
 * constexpr Integer Nx = 20, Ny = 20, Nz = 20;
 * auto foo = [](auto... indices) { };
 * ThreadLoop< Nx, Ny, Nz >(thread, foo);
 * ```
 * 
 * is equivalent to:
 * ```
 * for ( int i = threadIdx.x; i < Nx; i += Bx )
 * {
 *    for ( int j = threadIdx.j; j < Ny; j += By )
 *    {
 *       for ( int k = 0; k < Nz; ++k )
 *       {
 *          foo( i, j, k );
 *       }
 *    }
 * }
 * ```
*/
template < size_t ... LoopDims, typename ThreadLayout, typename Lambda >
GENDIL_HOST_DEVICE GENDIL_INLINE
void ThreadLoop( const ThreadLayout & thread, Lambda && body )
{
   using bnds = std::index_sequence< LoopDims... >;
   details::ReverseThreadLoopImpl<0>( thread, std::forward< Lambda >( body ), bnds{} );
}

/**
 * @brief Arbitrary dimensional loop with last index running fastest. The first
 * ThreadLayout::Dim loop dimensions are parallelized by the threads.
 * 
 * @details example
 * ```
 * constexpr Integer Bx = 10, By = 10;
 * ThreadLayout< Bx, By > thread;
 * 
 * constexpr Integer Nx = 20, Ny = 20, Nz = 20;
 * using dims = std::index_sequence< Nx, Ny, Nz >;
 * auto foo = [](auto... indices) { };
 * ThreadLoop< dims >(thread, foo);
 * ```
 * 
 * is equivalent to:
 * ```
 * for ( int i = threadIdx.x; i < Nx; i += Bx )
 * {
 *    for ( int j = threadIdx.y; j < Ny; j += By )
 *    {
 *       for ( int k = 0; k < Nz; ++k )
 *       {
 *          foo( i, j, k );
 *       }
 *    }
 * }
 * ```
*/
template < typename LoopDimSequence, typename ThreadLayout, typename Lambda >
GENDIL_HOST_DEVICE GENDIL_INLINE
void ThreadLoop( const ThreadLayout & thread, Lambda && body )
{
   details::ReverseThreadLoopImpl<0>( thread, std::forward< Lambda >( body ), LoopDimSequence{} );
}

} // namespace gendil
