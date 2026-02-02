// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/View/view.hpp"
#include "gendil/Utilities/gettensorsize.hpp"

namespace gendil
{

template < typename Sizes, Integer ... Strides >
struct FixedStridedLayout
{
   using sizes = Sizes;

   static constexpr Integer rank = sizeof...(Strides);

   GENDIL_HOST_DEVICE
   constexpr FixedStridedLayout()
   {
      static_assert( ((Strides > 0) && ... && true), "All strides must be >= 1." );
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   constexpr Integer Offset( const std::tuple< Indices... > & index ) const
   {
      static_assert( sizeof...(Indices) == rank,  "Wrong number of indices." );
      return std::apply( []( auto ... indices ) constexpr { return ( 0 + ... + (Strides * indices) ); }, index );
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   constexpr Integer Offset( Indices... idx ) const
   {
      static_assert( sizeof...(Indices) == rank,  "Wrong number of indices." );
      return ( 0 + ... + (idx * Strides) );
   }

   template < Integer Index >
   GENDIL_HOST_DEVICE
   static constexpr Integer GetStride()
   {
      static_assert( Index < rank );
      return vseq_get_v< Index, Strides... >;
   }
};

namespace details
{

template < Integer... Sizes >
struct ComputeFIFOStride
{
   template < Integer N >
   static constexpr Integer Size()
   {
      return vseq_get_v< N, Sizes... >;
   }

   template < Integer N >
   static constexpr Integer Stride()
   {
      if constexpr (N == 0)
         return 1;
      else
         return Size<N-1>() * Stride<N-1>();
   }
};

template < Integer ... Sizes, size_t ... Is >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeFixedFIFOStridedLayout( std::index_sequence<Is...> )
{
   using FIFOStride = ComputeFIFOStride<Sizes...>;
   return FixedStridedLayout< std::index_sequence< Sizes... >, FIFOStride::template Stride<Is>()... >{};
}

} // namespace details

template < Integer ... Sizes >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeFixedFIFOStridedLayout()
{
   return details::MakeFixedFIFOStridedLayout< Sizes... >( std::make_index_sequence< sizeof...(Sizes) >{} );
}

template < Integer ... Sizes >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeFixedFIFOStridedLayout( std::index_sequence<Sizes...> )
{
   return MakeFixedFIFOStridedLayout<Sizes...>();
}

template < typename Container, typename Sizes, Integer ... Strides >
using FixedStridedView = View< Container, FixedStridedLayout< Sizes, Strides... > >;

template < Integer... Sizes, typename T >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeFixedFIFOView( T *data )
{
   return MakeView( data, MakeFixedFIFOStridedLayout< Sizes... >() );
}

template < typename T, Integer... Sizes >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeFixedFIFOView( T *data, std::index_sequence<Sizes...> )
{
   return MakeView( data, MakeFixedFIFOStridedLayout< Sizes... >() );
}

template < typename T, Integer... Sizes >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeStaticFIFOView()
{
   constexpr size_t Size = Product( Sizes... );
   return MakeView( StaticContainer< T, Size >{}, MakeFixedFIFOStridedLayout< Sizes... >() );
}

template < typename T, Integer... Sizes >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeStaticFIFOView( std::index_sequence<Sizes...> )
{
   constexpr size_t Size = Product( Sizes... );
   return MakeView( StaticContainer< T, Size >{}, MakeFixedFIFOStridedLayout< Sizes... >() );
}

template < typename Container, typename Sizes, Integer ... Strides >
GENDIL_HOST_DEVICE GENDIL_INLINE
FixedStridedView< Container, Sizes, Strides ... > & operator+=(
   FixedStridedView< Container, Sizes, Strides ... > & x,
   const FixedStridedView< Container, Sizes, Strides ... > & y)
{
   UnitLoop< Sizes >( [&]( auto ... indices )
   {
      x( indices... ) += y( indices... );
   });
   return x;
}

} // namespace gendil
