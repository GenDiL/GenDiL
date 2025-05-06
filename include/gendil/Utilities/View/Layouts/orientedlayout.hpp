// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/View/view.hpp"
#include "stridedlayout.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"

namespace gendil {

/**
 * @brief Similar to a strided layout, however allow negative stride to convey backward stride.
 * 
 * @tparam Rank The rank of the layout.
 */
template < size_t Rank >
struct OrientedLayout
{
    static constexpr size_t rank = Rank;
    size_t offset;
    int strides[ Rank ];

    template < typename... StrideSizePairs >
    GENDIL_HOST_DEVICE
    constexpr OrientedLayout( StrideSizePairs... pairs ) :
        offset{ ( ( pairs.first > 0 ? 0 : static_cast< size_t >( -pairs.first ) * (pairs.second-1) ) + ... ) },
        strides{ pairs.first... }
    {}

    template < typename... Indices >
    GENDIL_HOST_DEVICE
    constexpr size_t Offset( std::tuple< Indices... > & tuple ) const
    {
        static_assert(
            sizeof...(Indices) == Rank,
            "Wrong number of arguments." );
        return Offset( tuple, std::make_index_sequence< Rank >{} );
    }

    template < typename... Indices >
    GENDIL_HOST_DEVICE
    constexpr size_t Offset( Indices... idx ) const
    {
        static_assert(
            sizeof...(Indices) == Rank,
            "Wrong number of arguments." );
        auto tuple = std::make_tuple( idx... );
        return Offset( tuple );
    }

    template < typename Tuple, size_t... Is >
    GENDIL_HOST_DEVICE
    constexpr size_t Offset( Tuple && idx, std::index_sequence< Is... > ) const
    {
        return ( offset + ... + ( std::get< Is >( idx ) * strides[ Is ] ) );
    }
};

namespace detail {

    template < Integer Dim >
    GENDIL_HOST_DEVICE
    std::array< int, Dim > ComputedOrientedStrides(
        std::array< size_t, Dim > const & sizes,
        Permutation< Dim > const & orientations )
    {
        std::array< int, Dim > strides;
        size_t current_stride = 1;
        for (size_t i = 0; i < Dim; i++)
        {
            for (size_t j = 0; j < Dim; j++)
            {
                const int orientation = orientations( j );
                const size_t permutation = orientation > 0 ? orientation : -orientation;
                const int sign = orientation > 0 ? 1 : -1;
                if ( permutation == ( i+1 ) )
                {
                    strides[ j ] = sign * current_stride;
                    current_stride *= sizes[ j ];
                    break;
                }                
            }
        }
        return strides;
    }


    template < size_t dim, size_t... Is >
    GENDIL_HOST_DEVICE
    constexpr auto OrientationAndSizesToOrientedStridesAndSizes(
        std::array< size_t, dim > const & sizes,
        std::array< int, dim > const & oriented_strides,
        std::index_sequence< Is... > )
    {
        return std::make_tuple( std::make_pair( oriented_strides[ Is ], sizes[ Is ] )... );
    }

    template < size_t dim >
    GENDIL_HOST_DEVICE
    constexpr auto OrientationAndSizesToOrientedStridesAndSizes(
        std::array< size_t, dim > const & sizes,
        Permutation< dim > const & orientation )
    {
        auto oriented_strides = ComputedOrientedStrides( sizes, orientation );
        return OrientationAndSizesToOrientedStridesAndSizes( sizes, oriented_strides, std::make_index_sequence< dim >{} );
    }

}

template < Integer Rank, typename T = Real >
using OrientedView = View< PointerContainer< T >, OrientedLayout< Rank > >;

template < size_t dim >
GENDIL_HOST_DEVICE
constexpr auto MakeOrientedLayout(
    std::array< size_t, dim > const & sizes,
    Permutation< dim > const & orientation )
{
    return std::make_from_tuple< OrientedLayout< dim > >( detail::OrientationAndSizesToOrientedStridesAndSizes( sizes, orientation ) );
}

template < typename T, size_t dim >
GENDIL_HOST_DEVICE
auto MakeOrientedView(
    T* const data,
    std::array< size_t, dim > const & sizes,
    Permutation< dim > const & orientation )
{
    return MakeView( data, MakeOrientedLayout( sizes, orientation ) );
}

template < typename T, size_t dim >
GENDIL_HOST_DEVICE
auto MakeOrientedView( HostDevicePointer< T > const & data, std::array< size_t, dim > const & sizes, Permutation< dim > const & orientation )
{
    return MakeView( data, MakeOrientedLayout( sizes, orientation ) );
}

}
