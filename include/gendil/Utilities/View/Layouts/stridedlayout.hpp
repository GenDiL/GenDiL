// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/View/view.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"

namespace gendil
{

template < Integer Rank >
struct StridedLayout
{
    static constexpr Integer rank = Rank;
    GlobalIndex strides[ Rank ];

    template < typename... Strides >
    GENDIL_HOST_DEVICE
    constexpr StridedLayout( GlobalIndex first_stride, Strides... rest_strides ) :
        strides{ first_stride, rest_strides... }
    {}

    template < typename... Indices >
    GENDIL_HOST_DEVICE
    constexpr GlobalIndex Offset( std::tuple< Indices... > & idx ) const
    {
        static_assert(
            sizeof...(Indices) == Rank,
            "Wrong number of arguments." );
        return Offset( idx, std::make_index_sequence< Rank >{} );
    }

    template < typename Tuple, Integer... Is >
    GENDIL_HOST_DEVICE
    constexpr GlobalIndex Offset( Tuple && idx, std::index_sequence< Is... > ) const
    {
        return ( ( std::get< Is >( idx ) * strides[ Is ] ) + ... );
    }

    template < typename... Indices >
    GENDIL_HOST_DEVICE
    constexpr GlobalIndex Offset( Indices... idx ) const
    {
        static_assert(
            sizeof...(Indices) == Rank,
            "Wrong number of arguments." );
        auto tuple = std::make_tuple( std::forward< Indices >( idx ) ... );
        return Offset( tuple );
    }
};

namespace detail {

    template < Integer Rank, typename IndexType >
    GENDIL_HOST_DEVICE
    constexpr GlobalIndex ComputeFIFOStride( std::array< IndexType, Rank > const & sizes, size_t n )
    {
        GlobalIndex prod = 1;
        for (GlobalIndex i = 0; i < n; i++)
        {
            prod *= sizes[ i ];
        }    
        return prod;
    }

    template < Integer Rank, typename IndexType, Integer... Is >
    GENDIL_HOST_DEVICE
    auto MakeFIFOStridedLayout(
        std::array< IndexType, Rank > const & sizes,
        std::index_sequence< Is... > )
    {
        return StridedLayout< Rank >( ComputeFIFOStride( sizes, Is )... );
    }
}

template < Integer Rank, typename IndexType >
GENDIL_HOST_DEVICE
constexpr auto MakeFIFOStridedLayout( std::array< IndexType, Rank > const & sizes )
{
    return detail::MakeFIFOStridedLayout( sizes, std::make_index_sequence< Rank >{} );
}

template < Integer Rank, typename T = Real >
using StridedView = View< PointerContainer< T >, StridedLayout< Rank > >;
template < Integer Rank, typename T = Real >
using HostDeviceStridedView = View< HostDevicePointer< T >, StridedLayout< Rank > >;

template < typename T, typename IndexType, Integer dim >
GENDIL_HOST_DEVICE
auto MakeFIFOView(
    T* const data,
    std::array< IndexType, dim > const & sizes )
{
    return MakeView( data, MakeFIFOStridedLayout( sizes ) );
}

template < typename T, typename IndexType, Integer dim >
GENDIL_HOST_DEVICE
auto MakeFIFOView( const HostDevicePointer< T > & data, std::array< IndexType, dim > const & sizes )
{
    return MakeView( data, MakeFIFOStridedLayout( sizes ) );
}

template < typename T, typename IndexType, typename... Args >
GENDIL_HOST_DEVICE
auto MakeFIFOView(
    T* const data,
    IndexType const & first_size,
    Args const & ... rest_sizes )
{
    static constexpr Integer rank = sizeof...(Args) + 1;
    return MakeFIFOView( data, std::array< IndexType, rank>{ first_size, rest_sizes... } );
}

template < typename T, typename... Args >
GENDIL_HOST_DEVICE
auto MakeFIFOView( const HostDevicePointer< T > & data, Args const & ... sizes )
{
    constexpr Integer rank = sizeof...(Args);
    return MakeFIFOView( data, std::array< GlobalIndex, rank>{ sizes... } );
}

template < typename T, typename... Args >
GENDIL_HOST_DEVICE
auto MakeIndirectedFIFOView( const T & data, const HostDevicePointer< const int > & indirections, Args const & ... sizes )
{
    constexpr Integer rank = sizeof...(Args);
    using Container = IndirectedContainer< T >;
    return MakeView( Container{ data, indirections }, MakeFIFOStridedLayout( std::array< GlobalIndex, rank>{ (GlobalIndex)sizes... } ) );
}

namespace detail {

    template < Integer Dim >
    GENDIL_HOST_DEVICE
    std::array< GlobalIndex, Dim > ComputedPermutedStrides(
        std::array< GlobalIndex, Dim > const & sizes,
        Permutation< Dim > const & permutations )
    {
        std::array< GlobalIndex, Dim > strides{};
        GlobalIndex current_stride = 1;
        for (GlobalIndex i = 0; i < Dim; i++)
        {
            for (GlobalIndex j = 0; j < Dim; j++)
            {
                if ( permutations( j ) == ( i+1 ) )
                {
                    strides[ j ] = current_stride;
                    current_stride *= sizes[ j ];
                    break;
                }                
            }
        }
        return strides;
    }

    template < Integer dim, Integer... Is >
    GENDIL_HOST_DEVICE
    constexpr auto PermutationAndSizesToStridesAndSizes(
        std::array< GlobalIndex, dim > const & sizes,
        std::array< GlobalIndex, dim > const & permuted_strides,
        std::index_sequence< Is... > )
    {
        return std::make_tuple( permuted_strides[ Is ]... );
    }

    template < Integer dim >
    GENDIL_HOST_DEVICE
    constexpr auto PermutationAndSizesToStridesAndSizes(
        std::array< GlobalIndex, dim > const & sizes,
        Permutation< dim > const & permutation )
    {
        auto permuted_strides = ComputedPermutedStrides( sizes, permutation );
        return PermutationAndSizesToStridesAndSizes( sizes, permuted_strides, std::make_index_sequence< dim >{} );
    }

}

template < Integer dim >
GENDIL_HOST_DEVICE
constexpr auto MakeStridedLayout(
    std::array< GlobalIndex, dim > const & sizes,
    Permutation< dim > const & permutation )
{
    return std::make_from_tuple< StridedLayout< dim > >( detail::PermutationAndSizesToStridesAndSizes( sizes, permutation ) );
}

template < typename T, Integer dim >
GENDIL_HOST_DEVICE
auto MakeStridedView(
    T* const data,
    std::array< GlobalIndex, dim > const & sizes,
    Permutation< dim > const & permutation )
{
    return MakeView( data, MakeStridedLayout( sizes, permutation ) );
}

template < typename T, Integer dim >
GENDIL_HOST_DEVICE
constexpr auto MakeStridedLayout( HostDevicePointer< T > const & data, std::array< GlobalIndex, dim > const & sizes, Permutation< dim > const & permutation )
{
    return MakeView( data, MakeStridedLayout( sizes, permutation ) );
}

}
