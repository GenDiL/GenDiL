// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#if defined( GENDIL_USE_MFEM )
#include <mfem.hpp>
#include "general/forall.hpp"
#endif

namespace gendil
{

namespace details
{
   template < size_t... BlockDims, size_t... I >
   constexpr size_t ComputeTotalThreadsInBlock( std::index_sequence< I... > )
   {
      using dims = std::index_sequence< BlockDims... >;
      return (1 * ... * seq_get_v< I, dims > );
   }

   template < size_t End, size_t ... BlockDims >
   constexpr size_t ComputeTotalThreadsInBlock()
   {
      return ComputeTotalThreadsInBlock< BlockDims...>( std::make_index_sequence<End>{} );
   }

   template < Integer Back, size_t ... BlockDims, size_t N >
   GENDIL_HOST_DEVICE
   inline void ComputeThreadIndex( std::array< GlobalIndex, N > & ThreadIndex, GlobalIndex index )
   {
      if constexpr ( Back == 1 )
      {
         ThreadIndex[ 0 ] = index;
      }
      else if constexpr ( Back == 2 )
      {
         constexpr GlobalIndex NumThreads = vseq_get_v< 0, BlockDims... >;
         ThreadIndex[ 1 ] = index / NumThreads;
         ThreadIndex[ 0 ] = index - NumThreads * ThreadIndex[ 1 ];
      }
      else
      {
         constexpr GlobalIndex NumThreads = ComputeTotalThreadsInBlock< Back, BlockDims... >();
         ThreadIndex[ Back-1 ] = index / NumThreads;
         
         index -= NumThreads * ThreadIndex[ Back-1 ];

         ComputeThreadIndex< Back-1, BlockDims... >( ThreadIndex, index );
      }
   }

#ifdef GENDIL_USE_DEVICE
   template <typename Lambda> __global__ static
   void GridLoop(const int n, Lambda body)
   {
      for (int k = blockIdx.x; k < n; k += gridDim.x) { body( k ); }
   }
#endif
} // namespace details

/**
 * @brief Abstracts thread blocks to arbitrary number of dimensions.
 *
 * @details An instance of KernelContext should be thread local as the device
 * global variables threadIdx and blockDim are used to compute the N-dimensional
 * thread index.
 *
 * @tparam BlockDims the dims of the block
*/
template < size_t ... BlockDims >
class ThreadBlockLayout;

/**
 * @brief Abstraction of a single thread. In the CUDA/HIP threading model,
 * threads are always organized in thread blocks (which are typically multiples
 * of the warp size). The single thread ThreadBlockLayout<> ignores the
 * underlying thread block and always returns 0 for the thread index.
*/
template <>
class ThreadBlockLayout<>
{
public:
   static constexpr size_t thread_block_dim = 0; // TODO rename to thread_block_dim
   using block_dimensions = std::index_sequence< >;

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return 1;
   }

   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      return 1;
   }

   /**
    * @brief Since this thread block is zero-dimensional, no value of `Index` is
    * logical. In debug mode, this function throws an error, otherwise returns 1
    * for any `Index`. This function exists in compatibility with the other
    * ThreadBlockLayout<...> classes.
   */
   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      Assert(false, "This thread block is zero-dimensional (one thread), GetBlockDim<Index> is not logical for any Index!");
      return GetBlockDim();
   }

   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetThreadIndex()
   {
      return 0;
   }

   /**
    * @brief Since this thread block is zero-dimensional, no value of `Index` is
    * logical. In debug mode, this function throws an error, otherwise returns 0
    * for any `Index`. This function exists in compatibility with the other
    * ThreadBlockLayout<...> classes.
   */
   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetThreadIndex()
   {
      Assert(false, "This thread block is zero-dimensional (one thread), GetThreadIndex<Index> is not logical for any Index!");
      return GetThreadIndex();
   }

   /**
    * @brief forall loop. In CUDA/HIP this may be implemented using blocks with block
    * size greater than 1, however, GetBlockDim() always returns 1 and
    * GetThreadIndex() always returns 0.
   */
   template < typename Lambda >
   static inline void BlockLoop( GlobalIndex n, Lambda && body )
   {
   // #if defined( GENDIL_USE_MFEM )
   //    mfem::forall( n, std::forward< Lambda >( body ) );
   // #else
      #pragma omp parallel for
      for ( GlobalIndex i = 0; i < n; ++i )
      {
         body(i);
      }
   // #endif
   }

   /**
    * @brief Since this thread block consists of a single thread, this function
    * does nothing. In CUDA/HIP, the ThreadLayout may be implemented using
    * blocks with block size greater than 1, however, the threads represented by
    * this `ThreadBlockLayout<>` are assumed independant and therefore
    * synchronization of the CUDA/HIP thread block is not logical.
   */
   GENDIL_HOST_DEVICE
   inline void Synchronize() const
   {
      // does not actually Synchronize
   }
};

#ifdef GENDIL_USE_DEVICE // implementation of GPU thread block layouts

template < size_t BlockDim >
class ThreadBlockLayout< BlockDim >
{
public:
   static constexpr size_t thread_block_dim = 1;
   using block_dimensions = std::index_sequence< BlockDim>;

   GENDIL_DEVICE
   ThreadBlockLayout()
   {
   #ifdef GENDIL_DEVICE_CODE
      Assert( BlockDim <= blockDim.x, "Device kernel launched with an insufficient number of threads." );
   #elif !defined(NDEBUG)
      std::cerr << "Initializing ThreadBlockLayout<" << BlockDim << "> on CPU... did you mean to do this?" << std::endl;
   #endif
   }

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return BlockDim;
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      static_assert( Index == 0, "1D Thread block has only one dimension." );
      return BlockDim;
   }

   template < size_t Index >
   GENDIL_DEVICE
   GlobalIndex GetThreadIndex() const
   {
      static_assert( Index == 0, "1D Thread block has only one thread index." );
      return threadIdx.x;
   }

   GENDIL_DEVICE
   GlobalIndex GetLinearThreadIndex() const
   {
      return threadIdx.x;
   }

   template < typename Lambda >
   static inline void BlockLoop( GlobalIndex n, Lambda && body )
   {
   #if defined( GENDIL_USE_DEVICE )
      dim3 gridDim(n);
      dim3 blockDim(BlockDim);
      size_t sharedMemSize = 0;
      Stream_t stream = 0;    // Use default stream

      details::GridLoop<<< gridDim, blockDim, sharedMemSize, stream >>>( n, body );
   #elif defined( GENDIL_USE_MFEM )
      mfem::forall_2D( n, BlockDim, 1, std::forward< Lambda >( body ) );
   #else
      static_assert(false, "ThreadBlockLayout< ... >::BlockLoop() not yet implemented.");
   #endif
   }

   GENDIL_DEVICE
   inline void Synchronize() const
   {
      GENDIL_SYNC_THREADS();
   }
};

template < size_t BlockDimX, size_t BlockDimY >
class ThreadBlockLayout< BlockDimX, BlockDimY >
{
public:
   static constexpr size_t thread_block_dim = 2;
   using block_dimensions = std::index_sequence< BlockDimX, BlockDimY >;
   
   GENDIL_DEVICE
   ThreadBlockLayout()
   {
      Assert( BlockDimX <= blockDim.x && BlockDimY <= blockDim.y, "Device kernel launched with an insufficient number of threads." );
   }

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return BlockDimX * BlockDimY;
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      static_assert( Index == 0 || Index == 1, "2D Thread block has only two dimensions." );
      if constexpr ( Index == 0 )
         return BlockDimX;
      else // Index == 1
         return BlockDimY;
   }

   template < size_t Index >
   GENDIL_DEVICE
   GlobalIndex GetThreadIndex() const
   {
      static_assert( Index == 0 || Index == 1, "2D Thread block has only two dimensions." );
      if constexpr ( Index == 0 )
         return threadIdx.x;
      else // Index == 1
         return threadIdx.y;
   }

   GENDIL_DEVICE
   GlobalIndex GetLinearThreadIndex() const
   {
      return threadIdx.x + BlockDimX * threadIdx.y;
   }

   template < typename Lambda >
   static inline void BlockLoop( GlobalIndex n, Lambda && body )
   {
   #if defined( GENDIL_USE_DEVICE )
      dim3 gridDim(n);
      dim3 blockDim(BlockDimX, BlockDimY);
      size_t sharedMemSize = 0;
      Stream_t stream = 0;    // Use default stream

      details::GridLoop<<< gridDim, blockDim, sharedMemSize, stream >>>( n, std::forward< Lambda >(body) );
   #elif defined( GENDIL_USE_MFEM )
      mfem::forall_2D( n, BlockDimX, BlockDimY, std::forward< Lambda >( body ) );
   #else
      static_assert(false, "ThreadBlockLayout< ... >::BlockLoop() not yet implemented.");
   #endif
   }

   GENDIL_DEVICE
   inline void Synchronize() const
   {
      GENDIL_SYNC_THREADS();
   }
};

template < size_t BlockDimX, size_t BlockDimY, size_t BlockDimZ >
class ThreadBlockLayout< BlockDimX, BlockDimY, BlockDimZ >
{
public:
   static constexpr size_t thread_block_dim = 3;
   using block_dimensions = std::index_sequence< BlockDimX, BlockDimY, BlockDimZ >;

   GENDIL_DEVICE
   ThreadBlockLayout()
   {
      Assert( BlockDimX <= blockDim.x && BlockDimY <= blockDim.y && BlockDimZ <= blockDim.z, "Device kernel launched with an insufficient number of threads." );
   }

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return BlockDimX * BlockDimY * BlockDimZ;
   }

   template < size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      static_assert( Index == 0 || Index == 1 || Index == 2, "3D Thread block has only three dimension." );
      if constexpr ( Index == 0 )
         return BlockDimX;
      else if constexpr ( Index == 1 )
         return BlockDimY;
      else // Index == 2
         return BlockDimZ;
   }

   template < size_t Index >
   GENDIL_DEVICE
   GlobalIndex GetThreadIndex() const
   {
      static_assert( Index == 0 || Index == 1 || Index == 2, "3D Thread block has only three thread indices." );
      if constexpr ( Index == 0 )
         return threadIdx.x;
      else if constexpr ( Index == 1 )
         return threadIdx.y;
      else // Index == 2
         return threadIdx.z;
   }

   GENDIL_DEVICE
   GlobalIndex GetLinearThreadIndex() const
   {
      return threadIdx.x + BlockDimX * (threadIdx.y + BlockDimY * threadIdx.z);
   }

   template < typename Lambda >
   static inline void BlockLoop( GlobalIndex n, Lambda && body )
   {
   #if defined( GENDIL_USE_DEVICE )
      dim3 gridDim(n);
      dim3 blockDim(BlockDimX, BlockDimY, BlockDimZ);
      size_t sharedMemSize = 0;
      Stream_t stream = 0; // Use default stream

      details::GridLoop<<< gridDim, blockDim, sharedMemSize, stream >>>( n, body );
   #elif defined( GENDIL_USE_MFEM )
      mfem::forall_3D( n, BlockDimX, BlockDimY, BlockDimZ, std::forward< Lambda >( body ) );
   #else
      static_assert(false, "ThreadBlockLayout< ... >::BlockLoop() not yet implemented.");
   #endif
   }

   GENDIL_DEVICE
   inline void Synchronize() const
   {
      GENDIL_SYNC_THREADS();
   }
};

template < size_t ... BlockDims >
class ThreadBlockLayout
{
public:
   static constexpr size_t thread_block_dim = sizeof...(BlockDims);
   using block_dimensions = std::index_sequence< BlockDims ... >;

   GENDIL_DEVICE
   ThreadBlockLayout()
   {
      static_assert( thread_block_dim > 3, "ThreadBlockLayout has an illogical number of dimensions" );
      Assert( (BlockDims * ... * 1) <= blockDim.x, "Device kernel launched with an insufficient number of threads." );
      details::ComputeThreadIndex< thread_block_dim, BlockDims... >( ThreadIndex, threadIdx.x );
   }

   GENDIL_HOST_DEVICE
   static constexpr size_t GetNumberOfThreads()
   {
      return (1 * ... * BlockDims);
   }

   template< size_t Index >
   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex GetBlockDim()
   {
      static_assert( Index < thread_block_dim, "Thread block dimension is greater than the order of the block.");
      return vseq_get_v< Index, BlockDims... >;
   }

   template < size_t Index >
   GENDIL_DEVICE
   constexpr GlobalIndex GetThreadIndex() const
   {
      static_assert( Index < thread_block_dim, "Thread block index dimension is greater than the order of the block." );
      return ThreadIndex[ Index ];
   }

   GENDIL_DEVICE
   GlobalIndex GetLinearThreadIndex() const
   {
      return threadIdx.x;
   }

   template < typename Lambda >
   static inline void BlockLoop( GlobalIndex n, Lambda && body )
   {
      // TODO Add a batch_size?
   #if defined( GENDIL_USE_DEVICE )
      dim3 gridDim(n);
      dim3 blockDim(GetNumberOfThreads());
      size_t sharedMemSize = 0;
      Stream_t stream = 0;    // Use default stream

      details::GridLoop<<< gridDim, blockDim, sharedMemSize, stream >>>( n, body );
   #elif defined( GENDIL_USE_MFEM )
      mfem::forall_2D( n, GetNumberOfThreads(), 1, std::forward< Lambda >( body ) );
   #else
      static_assert(false, "ThreadBlockLayout< ... >::BlockLoop() not implemented!");
   #endif
   }

   GENDIL_DEVICE
   inline void Synchronize() const
   {
      GENDIL_SYNC_THREADS();
   }

private:
   std::array< GlobalIndex, thread_block_dim > ThreadIndex;
};

#endif

} // namespace gendil