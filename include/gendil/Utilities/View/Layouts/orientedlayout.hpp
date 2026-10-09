// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Compact signed-stride mappings for oriented tensor axes. */

#include "gendil/Utilities/View/view.hpp"
#include "gendil/Utilities/View/extents.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"
#include "gendil/Meshes/Connectivities/Orientations/tensorproductorientation.hpp"

#include <cstdint>
#include <limits>

namespace gendil {

namespace detail {

/** Check each pair separately so the constructor constraint folds a concept. */
template < class Pair >
concept StrideSizePair = requires( Pair pair )
{
    pair.first;
    pair.second;
};

} // namespace detail

/**
 * @brief Mapping-only affine layout with signed strides for reversed axes.
 *
 * Signed strides encode axis directions, while offset addresses native zero.
 * Sizes are setup inputs only; no shape policy is stored or queried.
 *
 * @tparam Rank The rank of the layout.
 */
template < size_t Rank >
struct OrientedLayout
{
    static constexpr size_t rank = Rank;
    size_t offset{};
    int strides[Rank == 0 ? 1 : Rank]{};

    GENDIL_HOST_DEVICE
    constexpr OrientedLayout() = default;

    template < typename... StrideSizePairs >
        requires ( sizeof...( StrideSizePairs ) == Rank && Rank > 0 &&
                   ( detail::StrideSizePair< StrideSizePairs > && ... ) )
    GENDIL_HOST_DEVICE
    constexpr OrientedLayout( StrideSizePairs... pairs ) :
        strides{ pairs.first... }
    {
        ( [&] {
            const auto length = detail::CheckedExtentCast< size_t >( pairs.second );
            GENDIL_ASSERT( length > 0, "Oriented layout dimensions must be positive." );
            if ( pairs.first < 0 )
            {
                const auto magnitude = static_cast< size_t >( -static_cast< int64_t >( pairs.first ) );
                GENDIL_ASSERT( length - 1 <= std::numeric_limits< size_t >::max() / magnitude,
                                      "Oriented layout reverse offset overflows." );
                const auto shift = magnitude * ( length - 1 );
                GENDIL_ASSERT( shift <= std::numeric_limits< size_t >::max() - offset,
                                      "Oriented layout base offset overflows." );
                offset += shift;
            }
        }(), ... );
    }

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
        // Unsigned arithmetic avoids signed intermediate overflow. Valid
        // indices produce the representable address established at setup;
        // negative-stride contributions use unsigned modular subtraction.
        return ( offset + ... + ( static_cast< size_t >( std::get< Is >( idx ) ) *
                                  static_cast< size_t >( strides[ Is ] ) ) );
    }
};

namespace detail {

// Keep per-axis setup separate so it can be inlined independently.
template < size_t Axis, size_t Rank, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr void SetNativeOrientedAxis(
    OrientedLayout< Rank > & result,
    const std::array< size_t, Rank > & sizes,
    const std::array< size_t, Rank > & reference_strides,
    const Orientation & orientation )
{
    const LocalIndex mapped = GetStructuredOrientationAxis< Axis >( orientation );
    const size_t source = static_cast< size_t >( mapped > 0 ? mapped - 1 : -mapped - 1 );
    const int stride = static_cast< int >( reference_strides[source] );
    result.strides[Axis] = mapped > 0 ? stride : -stride;
    if ( mapped < 0 )
        result.offset += reference_strides[source] * ( sizes[Axis] - 1 );
}

// Explicit expansion keeps runtime-factor setup inlined in measured CPU kernels.
template < size_t Rank, class Orientation, size_t... Axis >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr void SetNativeOrientedAxes(
    OrientedLayout< Rank > & result,
    const std::array< size_t, Rank > & sizes,
    const std::array< size_t, Rank > & reference_strides,
    const Orientation & orientation,
    std::index_sequence< Axis... > )
{
    ( SetNativeOrientedAxis< Axis >( result, sizes, reference_strides, orientation ), ... );
}

} // namespace detail

template < Integer Rank, typename T = Real >
using OrientedView = View< PointerContainer< T >, OrientedLayout< Rank > >;

/**
 * Construct a signed-stride mapping from positive native-axis sizes.
 * Orientations may be runtime permutations, static identities, or products
 * of those representations. Construction stores no orientation or shape data.
 * Sizes and offsets must be representable; these internal preconditions are
 * checked only in CPU debug builds, not in release or device execution.
 */
template < size_t dim, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeOrientedLayout(
    std::array< size_t, dim > const & sizes,
    const Orientation & orientation )
{
    static_assert( orientation_dimension_v< Orientation > == dim );
    // Connectivity construction validates the orientation. Sizes and
    // orientation metadata are not retained after computing the mapping.
    OrientedLayout< dim > result;
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
    for ( const auto length : sizes )
        GENDIL_ASSERT( length > 0, "Oriented layout dimensions must be positive." );
#endif
    std::array< size_t, dim > reference_sizes{};
    ConstexprLoop< dim >( [&]( auto axis )
    {
        const LocalIndex mapped = detail::GetStructuredOrientationAxis< axis >( orientation );
        const size_t source = static_cast< size_t >( mapped > 0 ? mapped - 1 : -mapped - 1 );
        reference_sizes[source] = sizes[axis];
    } );
    std::array< size_t, dim > reference_strides{};
    size_t product = 1;
    for ( size_t axis = 0; axis < dim; ++axis )
    {
        GENDIL_ASSERT( product <= static_cast< size_t >( std::numeric_limits< int >::max() ),
                              "Oriented layout stride does not fit in int." );
        reference_strides[axis] = product;
        GENDIL_ASSERT( reference_sizes[axis] <= std::numeric_limits< size_t >::max() / product,
                              "Oriented layout extent product overflows." );
        product *= reference_sizes[axis];
    }
    detail::SetNativeOrientedAxes( result, sizes, reference_strides, orientation,
                                  std::make_index_sequence< dim >{} );
    // Each displacement is a term in the representable contiguous span above;
    // their sum cannot exceed product - 1.
    return result;
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

} // namespace gendil
