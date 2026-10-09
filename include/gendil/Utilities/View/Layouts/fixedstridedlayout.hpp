// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/View/view.hpp"
#include "gendil/Utilities/View/Layouts/strides.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/get.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

namespace gendil
{

/** Fixed tensor dimensions and strides are positive; rank zero is a scalar. */
template < typename Sizes, Integer ... Strides >
struct FixedStridedLayout
{
   using sizes = Sizes;

   static constexpr Integer rank = sizeof...(Strides);
   static_assert( Sizes::size() == rank,
      "Fixed strided layout requires one size per stride." );
   static_assert(
      []< size_t... Axis >( std::index_sequence< Axis... > )
      {
         return ( ( seq_get_v< Axis, Sizes > > 0 ) && ... );
      }( std::make_index_sequence< sizeof...( Strides ) >{} ),
      "Fixed strided layout extents must be positive." );

   template < size_t Axis >
      requires ( Axis < static_cast< size_t >( rank ) )
   GENDIL_HOST_DEVICE
   static constexpr auto Get( AxisTag< Axis > )
   {
      return std::integral_constant<
         GlobalIndex,
         static_cast< GlobalIndex >( seq_get_v< Axis, Sizes > ) >{};
   }

   GENDIL_HOST_DEVICE
   constexpr FixedStridedLayout()
   {
      constexpr auto shape = []< size_t... Axis >( std::index_sequence< Axis... > )
      {
         static_assert(
            ( std::in_range< GlobalIndex >( seq_get_v< Axis, Sizes > ) && ... ),
            "Fixed strided layout extents must be representable in GlobalIndex." );
         return std::array< GlobalIndex, rank >{
            static_cast< GlobalIndex >( seq_get_v< Axis, Sizes > )... };
      }( std::make_index_sequence< rank >{} );
      static_assert( detail::StaticExtentProductFits( shape ),
         "Fixed strided layout logical size overflows GlobalIndex." );
      static_assert( ((Strides > 0) && ... && true),
         "Fixed layouts require positive strides." );
      // Bound each term before arithmetic; validation is entirely type-known.
      constexpr bool span_fits = [&]
      {
         if constexpr ( rank > 0 )
         {
            constexpr std::array< GlobalIndex, rank > strides{ Strides... };
            GlobalIndex remaining = std::numeric_limits< Integer >::max();
            for ( size_t axis = 0; axis < rank; ++axis )
            {
               if ( shape[axis] == 0 || strides[axis] == 0 ||
                    shape[axis] - 1 > remaining / strides[axis] )
                  return false;
               remaining -= ( shape[axis] - 1 ) * strides[axis];
            }
         }
         return true;
      }();
      static_assert( span_fits,
         "Fixed strided layout offset is not representable in Integer." );
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

   template < size_t Axis >
      requires ( Axis < static_cast< size_t >( rank ) )
   GENDIL_HOST_DEVICE
   static constexpr auto Stride( AxisTag< Axis > )
   {
      return std::integral_constant<
         GlobalIndex,
         static_cast< GlobalIndex >(
            vseq_get_v< static_cast< Integer >( Axis ), Strides... > ) >{};
   }

   GENDIL_HOST_DEVICE
   static constexpr GlobalIndex Stride( const size_t axis )
   {
      constexpr std::array< GlobalIndex, rank > values{
         static_cast< GlobalIndex >( Strides )... };
      return values[axis];
   }
};

namespace details
{

template < Integer... Sizes >
struct ComputeFIFOStride
{
   static_assert( ( ( Sizes > 0 ) && ... ),
      "Fixed FIFO layout extents must be positive." );

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
      {
         constexpr Integer previous = Stride<N-1>();
         static_assert( Size<N-1>() <= std::numeric_limits<Integer>::max() / previous,
            "Fixed FIFO layout stride overflows Integer." );
         return Size<N-1>() * previous;
      }
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
   constexpr size_t Size = GetLogicalSize(
      StaticExtents< static_cast< GlobalIndex >( Sizes )... >{} );
   return MakeView( StaticContainer< T, Size >{}, MakeFixedFIFOStridedLayout< Sizes... >() );
}

template < typename T, Integer... Sizes >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeStaticFIFOView( std::index_sequence<Sizes...> )
{
   return MakeStaticFIFOView< T, Sizes... >();
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
