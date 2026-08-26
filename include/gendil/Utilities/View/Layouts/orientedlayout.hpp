// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/View/view.hpp"
#include "stridedlayout.hpp"
#include "gendil/Meshes/Connectivities/Orientations/tensorproductorientation.hpp"

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
    GENDIL_HOST_DEVICE GENDIL_INLINE
    constexpr size_t Offset( std::tuple< Indices... > & tuple ) const
    {
        static_assert(
            sizeof...(Indices) == Rank,
            "Wrong number of arguments." );
        return Offset( tuple, std::make_index_sequence< Rank >{} );
    }

    template < typename... Indices >
    GENDIL_HOST_DEVICE GENDIL_INLINE
    constexpr size_t Offset( Indices... idx ) const
    {
        static_assert(
            sizeof...(Indices) == Rank,
            "Wrong number of arguments." );
        auto tuple = std::make_tuple( idx... );
        return Offset( tuple );
    }

    template < typename Tuple, size_t... Is >
    GENDIL_HOST_DEVICE GENDIL_INLINE
    constexpr size_t Offset( Tuple && idx, std::index_sequence< Is... > ) const
    {
        return ( offset + ... + ( std::get< Is >( idx ) * strides[ Is ] ) );
    }
};

namespace detail {

    template < Integer Dim >
    GENDIL_HOST_DEVICE GENDIL_INLINE
    std::array< int, Dim > ComputedOrientedStrides(
        std::array< size_t, Dim > const & sizes,
        Permutation< Dim > const & orientations )
    {
        std::array< int, Dim > strides{};
        std::array< size_t, Dim > reference_sizes{};
        std::array< size_t, Dim > reference_strides{};
        for ( size_t native_axis = 0;
              native_axis < static_cast< size_t >( Dim );
              ++native_axis )
        {
            const int orientation = orientations( native_axis );
            const size_t reference_axis = static_cast< size_t >(
                orientation > 0 ? orientation - 1 : -orientation - 1 );
            reference_sizes[ reference_axis ] = sizes[ native_axis ];
        }
        size_t current_stride = 1;
        for ( size_t reference_axis = 0;
              reference_axis < static_cast< size_t >( Dim );
              ++reference_axis )
        {
            reference_strides[ reference_axis ] = current_stride;
            current_stride *= reference_sizes[ reference_axis ];
        }

        for ( size_t native_axis = 0;
              native_axis < static_cast< size_t >( Dim );
              ++native_axis )
        {
            const int orientation = orientations( native_axis );
            const size_t reference_axis = static_cast< size_t >(
                orientation > 0 ? orientation - 1 : -orientation - 1 );
            const int sign = orientation > 0 ? 1 : -1;
            strides[ native_axis ] =
                sign * static_cast< int >( reference_strides[ reference_axis ] );
        }
        return strides;
    }

    template < size_t Dim, size_t Rank >
    GENDIL_HOST_DEVICE GENDIL_INLINE
    constexpr std::array< int, Dim > ComputedOrientedStrides(
        const std::array< size_t, Dim > & sizes,
        const IdentityOrientation< Rank > & )
    {
        static_assert( Rank == Dim );
        std::array< int, Dim > strides{};
        size_t current_stride = 1;
        for ( size_t axis = 0; axis < Dim; ++axis )
        {
            strides[ axis ] = static_cast< int >( current_stride );
            current_stride *= sizes[ axis ];
        }
        return strides;
    }

    template < size_t Dim, class... Orientations >
    GENDIL_HOST_DEVICE GENDIL_INLINE
    std::array< int, Dim > ComputedOrientedStrides(
        const std::array< size_t, Dim > & sizes,
        const TensorProductOrientation< Orientations... > & orientation )
    {
        static_assert(
            TensorProductOrientation< Orientations... >::Dim == Dim,
            "Tensor-product orientation and layout dimensions must match." );

        std::array< size_t, Dim > reference_sizes{};
        gendil::ConstexprLoop< Dim >( [&] ( auto native_axis )
        {
            const auto mapped_axis =
                GetStructuredOrientationAxis< native_axis, 0 >( orientation );
            if constexpr (
                is_static_orientation_axis_v< decltype( mapped_axis ) > )
            {
                reference_sizes[ native_axis ] = sizes[ native_axis ];
            }
            else
            {
                const size_t reference_axis = static_cast< size_t >(
                    mapped_axis > 0 ? mapped_axis - 1 : -mapped_axis - 1 );
                reference_sizes[ reference_axis ] = sizes[ native_axis ];
            }
        } );
        std::array< size_t, Dim > reference_strides{};
        size_t current_stride = 1;
        for ( size_t reference_axis = 0;
              reference_axis < Dim;
              ++reference_axis )
        {
            reference_strides[ reference_axis ] = current_stride;
            current_stride *= reference_sizes[ reference_axis ];
        }

        std::array< int, Dim > strides{};
        gendil::ConstexprLoop< Dim >( [&] ( auto native_axis )
        {
            const auto mapped_axis =
                GetStructuredOrientationAxis< native_axis, 0 >( orientation );
            if constexpr (
                is_static_orientation_axis_v< decltype( mapped_axis ) > )
            {
                strides[ native_axis ] = static_cast< int >(
                    reference_strides[ native_axis ] );
            }
            else
            {
                const size_t reference_axis = static_cast< size_t >(
                    mapped_axis > 0 ? mapped_axis - 1 : -mapped_axis - 1 );
                strides[ native_axis ] =
                    ( mapped_axis > 0 ? 1 : -1 ) * static_cast< int >(
                        reference_strides[ reference_axis ] );
            }
        } );
        return strides;
    }


    template < size_t dim, size_t... Is >
    GENDIL_HOST_DEVICE GENDIL_INLINE
    constexpr auto OrientationAndSizesToOrientedStridesAndSizes(
        std::array< size_t, dim > const & sizes,
        std::array< int, dim > const & oriented_strides,
        std::index_sequence< Is... > )
    {
        return std::make_tuple( std::make_pair( oriented_strides[ Is ], sizes[ Is ] )... );
    }

    template < size_t dim, class Orientation >
    GENDIL_HOST_DEVICE GENDIL_INLINE
    constexpr auto OrientationAndSizesToOrientedStridesAndSizes(
        std::array< size_t, dim > const & sizes,
        const Orientation & orientation )
    {
        static_assert(
            orientation_dimension_v< Orientation > == dim,
            "Orientation and layout dimensions must match." );
        const auto oriented_strides =
            ComputedOrientedStrides( sizes, orientation );
        return OrientationAndSizesToOrientedStridesAndSizes(
            sizes,
            oriented_strides,
            std::make_index_sequence< dim >{} );
    }

}

template < Integer Rank, typename T = Real >
using OrientedView = View< PointerContainer< T >, OrientedLayout< Rank > >;

template < size_t dim, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeOrientedLayout(
    std::array< size_t, dim > const & sizes,
    const Orientation & orientation )
{
    return std::make_from_tuple< OrientedLayout< dim > >(
        detail::OrientationAndSizesToOrientedStridesAndSizes(
            sizes,
            orientation ) );
}

template < typename T, size_t dim, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto MakeOrientedView(
    T* const data,
    std::array< size_t, dim > const & sizes,
    const Orientation & orientation )
{
    return MakeView( data, MakeOrientedLayout( sizes, orientation ) );
}

template < typename T, size_t dim, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto MakeOrientedView(
    HostDevicePointer< T > const & data,
    std::array< size_t, dim > const & sizes,
    const Orientation & orientation )
{
    return MakeView( data, MakeOrientedLayout( sizes, orientation ) );
}

}
