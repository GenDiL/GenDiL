// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

namespace gendil
{

template < Integer NumThreads >
struct ThreadedDim
{
   static constexpr Integer NumberOfThreads = NumThreads;
};

struct SequentialDim
{
   static constexpr Integer NumberOfThreads = 1;
};

namespace details
{
   template < typename T >
   struct IsThreaded : std::false_type { };

   template < Integer NumThreads >
   struct IsThreaded<ThreadedDim<NumThreads>> : std::true_type { };

   template < typename T >
   struct __IsThreadingStrategy : std::false_type { };

   template < >
   struct __IsThreadingStrategy< SequentialDim > : std::true_type { };

   template < Integer N >
   struct __IsThreadingStrategy< ThreadedDim< N > > : std::true_type { };

   template < Integer End, typename ... Dims >
   GENDIL_HOST_DEVICE
   constexpr inline Integer ComputeTotalThreadsInBlock()
   {
      Integer NumThreads = 1;
      ConstexprLoop<End>([&] (auto i) constexpr
      {
         using SubDim = std::tuple_element_t<i, std::tuple< Dims ... >>;
         NumThreads *= SubDim::NumberOfThreads;
      });

      return NumThreads;
   }

   template < typename T >
   inline constexpr bool is_threaded_v = details::IsThreaded<T>::value;

   // ComputeStructuredSubIndices specialized to KernelContext
   template < Integer Back, typename ... Dims, Integer N >
   GENDIL_HOST_DEVICE
   inline void ComputeThreadIndices( GlobalIndex (&sub_inds)[N], Integer index )
   {
      if constexpr ( Back == 1 )
      {
         sub_inds[ 0 ] = index;
      }
      else if constexpr ( Back == 2 )
      {
         constexpr GlobalIndex NumThreads = std::tuple_element_t<0, std::tuple< Dims ... >>::NumberOfThreads;
         sub_inds[ 1 ] = index / NumThreads;
         sub_inds[ 0 ] = index - NumThreads * sub_inds[ 1 ];
      }
      else
      {
         constexpr GlobalIndex NumThreads = ComputeTotalThreadsInBlock< Back-1, Dims... >();
         sub_inds[ Back-1 ] = index / NumThreads;
         
         index -= NumThreads * sub_inds[ Back-1 ];

         ComputeThreadIndices< Back-1, Dims ... >( sub_inds, index );
      }
   }
} // namespace details

/**
 * @brief Represents a threading strategy (used together with Loop) for
 * N-dimensional block of operations which may be treated in parallel or
 * sequentially.
 *
 * @details An instance of KernelContext should be thread local as the threadIdx
 * is used on construction to compute the structured thread index.
 *
 * @tparam Dims each is either ThreadedDim or SequentialDim
*/
template < typename ... Dims >
class KernelContext
{
public:
   static constexpr Integer Dim = sizeof...(Dims);
   Real * SharedData; // pass around the shared memory pointer between device functions

   GENDIL_HOST_DEVICE
   KernelContext()
   {
      static_assert( sizeof...(Dims) > 0, "KernelContext with zero dimensions is not logical." );
      static_assert( (details::__IsThreadingStrategy<Dims>::value && ... && true), "Template Parameters to KernelContext must either StaticDim or ThreadDim< NumThreads >." );
   
   #ifdef GENDIL_DEVICE_CODE
      details::ComputeThreadIndices< Dim, Dims ... >(ThreadIndex, threadIdx.x );
   #else
      ConstexprLoop< Dim >([&](auto i) { ThreadIndex[ i ] = 0; });
   #endif
   }

   /// @brief returns the block dimension for dimension Index
   template< Integer Index >
   GENDIL_HOST_DEVICE
   static constexpr Integer GetBlockDim()
   {
      static_assert( 0 <= Index && Index < Dim, "Index is out of bounds for kernel context");
      using SubDim = std::tuple_element_t<Index, std::tuple< Dims ... > >;
      return SubDim::NumberOfThreads;
   }

   template < Integer Index >
   GENDIL_HOST_DEVICE
   static constexpr bool IsThreaded()
   {
      static_assert( 0 <= Index && Index < Dim, "Index is out of bounds for kernel context");
      return details::is_threaded_v< std::tuple_element_t<Index, std::tuple< Dims ... > > >;
   }

   template < Integer Index >
   GENDIL_HOST_DEVICE
   static constexpr bool IsSequential()
   {
      return not IsThreaded<Index>();
   }

   GENDIL_HOST_DEVICE
   static constexpr Integer GetNumberOfThreads() // <- FIXME: GetNumberOfThreadsInBlock ?
   {
      return ( 1 * ... * Dims::NumberOfThreads );
   }

   /// @brief When called inside a kernel, returns the thread index within the
   /// thread block.
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex GetThreadIndex( GlobalIndex subdim ) const
   {
      return ThreadIndex[ subdim ];
   }

   /// @brief When called inside a kernel, returns the thread index within the
   /// thread block.
   template < Integer subdim >
   GENDIL_HOST_DEVICE
   constexpr GlobalIndex GetThreadIndex() const
   {
      static_assert( 0 <= subdim && subdim < Dim, "subdim is out of bounds for kernel context");
      return ThreadIndex[ subdim ];
   }

private:
   GlobalIndex ThreadIndex[Dim];
};

} // namespace gendil
