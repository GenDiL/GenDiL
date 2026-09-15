// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Access flat buffers using matrix or tensor indices.
 *
 * These views let numerical code use coordinates such as (i,j) instead of
 * computing a position in a flat array at each access. A layout describes
 * where entries are stored; a view combines that layout with the data.
 *
 * For example, view an existing six-entry buffer as a 2-by-3 tensor:
 * @code
 * Real values[6]{};
 * auto tensor = MakeFIFOView(values, 2, 3);
 * tensor(1, 2) = 5; // Writes values[5]; no data is copied.
 * @endcode
 *
 * MakeFIFOView uses first-coordinate-fastest storage: tensor(0,j) and
 * tensor(1,j) are adjacent. MakeStridedView supports other axis orders.
 * For explicit spacing, such as padded storage, construct a StridedLayout
 * and combine it with data through MakeView.
 */

#include "gendil/Utilities/Loop/constexprloop.hpp"
#include "gendil/Utilities/View/view.hpp"
#include "gendil/Utilities/View/Layouts/strides.hpp"
#include "gendil/Meshes/Connectivities/Orientations/permutation.hpp"

namespace gendil
{

namespace detail {

/** Admit only the three supported extent representations with matching rank. */
template < Integer Rank, class Extents >
inline constexpr bool valid_strided_extents_v =
    std::same_as< Extents, NoExtents< Rank > > ||
    std::same_as< Extents, DynamicExtents< Rank > >;

template < Integer Rank, GlobalIndex... Sizes >
inline constexpr bool valid_strided_extents_v< Rank, StaticExtents< Sizes... > > =
    sizeof...( Sizes ) == Rank;

} // namespace detail

/**
 * @brief Describe where tensor entries are found in a linear container.
 *
 * A stride is the distance between entries when one coordinate increases by
 * one. For a 2-by-3 tensor with its first coordinate varying fastest, strides
 * {1,2} place entry (i,j) at i + 2*j. Offset returns this position; a View
 * uses it to access the container, whose semantics determine the result.
 *
 * In general, Offset computes @f$ \sum_a i_a s_a @f$, with strides measured
 * in container entries, not bytes. Gaps support padded storage; zero strides
 * make different coordinates refer to the same entry. For reversed axes,
 * use OrientedLayout rather than negative strides here.
 *
 * Shape is optional: indexing needs strides but not dimension lengths.
 * Supply Extents when consumers also need GetExtent/GetLogicalSize queries.
 * Shape does not change indexing, imply contiguous storage, or describe the
 * backing allocation's size. A rank-zero layout represents one scalar.
 *
 * @tparam Rank Number of tensor axes.
 * @tparam Extents NoExtents (default), StaticExtents, or DynamicExtents of Rank.
 * @pre Dimensions, strides, and resulting offsets fit in GlobalIndex.
 * Runtime construction checks are CPU-debug-only; release/device execution
 * trusts valid metadata and coordinates. Indexing performs no bounds checks.
 * A layout neither owns nor synchronizes the storage used by its view.
 */
template < Integer Rank, class Extents = NoExtents< Rank > >
    requires detail::valid_strided_extents_v< Rank, Extents >
struct StridedLayout
{
    static constexpr Integer rank = Rank;
    std::array< GlobalIndex, Rank > strides{}; ///< Increment per logical axis.
    [[no_unique_address]] Extents extents{}; ///< Optional shape, independent of strides.

    /** Expose one axis through GetExtent; unavailable for an unsized axis. */
    template < size_t Axis >
       requires has_extent_v< Axis, Extents >
    GENDIL_HOST_DEVICE
    constexpr decltype(auto) Get( AxisTag< Axis > ) const
    {
        return GetExtent< Axis >( extents );
    }

    /** Construct the scalar (rank-zero) layout, whose offset is zero. */
    GENDIL_HOST_DEVICE
    constexpr StridedLayout() requires ( Rank == 0 ) = default;

    /**
     * Construct a layout with explicit spacing and optional shape queries.
     * @code
     * StridedLayout<2, DynamicExtents<2>> layout{
     *    std::array{1, 2}, DynamicExtents<2>{2, 3}};
     * @endcode
     * CPU debug checks that the described tensor's offsets are representable.
     */
    template < detail::ExtentIndex Index >
    GENDIL_HOST_DEVICE
    constexpr StridedLayout(
        const std::array< Index, Rank > & input_strides,
        Extents input_extents )
        : extents( std::move( input_extents ) )
    {
        for ( size_t axis = 0; axis < Rank; ++axis )
            strides[axis] = detail::CheckedExtentCast< GlobalIndex >( input_strides[axis] );
        Validate();
    }

    /** Return the runtime stride for a compile-time-selected axis. */
    template < size_t Axis >
       requires ( Axis < static_cast< size_t >( Rank ) )
    GENDIL_HOST_DEVICE
    constexpr GlobalIndex Stride( AxisTag< Axis > ) const
    {
        return strides[Axis];
    }

    /** Return an axis stride; axis must be less than Rank. */
    GENDIL_HOST_DEVICE
    constexpr GlobalIndex Stride( const size_t axis ) const
    {
        return strides[axis];
    }

    /** Supply exactly Rank strides and default-construct the selected extent type. */
    template < detail::ExtentIndex... Strides >
       requires ( Rank > 0 && sizeof...( Strides ) == Rank )
    GENDIL_HOST_DEVICE
    constexpr StridedLayout( Strides... input ) :
        strides{ detail::CheckedExtentCast< GlobalIndex >( input )... }
    {
        Validate();
    }

    /** Flatten a tuple of Rank valid logical coordinates without bounds checks. */
    template < typename... Indices >
    GENDIL_HOST_DEVICE
    constexpr GlobalIndex Offset( std::tuple< Indices... > & idx ) const
    {
        static_assert(
            sizeof...(Indices) == Rank,
            "Wrong number of arguments." );
        return Offset( idx, std::make_index_sequence< Rank >{} );
    }

    /** Sum coordinate/stride products for the selected tuple axes. */
    template < typename Tuple, Integer... Is >
    GENDIL_HOST_DEVICE
    constexpr GlobalIndex Offset( Tuple && idx, std::index_sequence< Is... > ) const
    {
        return ( GlobalIndex{ 0 } + ... +
                 ( std::get< Is >( idx ) * strides[ Is ] ) );
    }

    /** Flatten Rank logical coordinates; the empty coordinate list maps to zero. */
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

private:
    /** Check internal span preconditions, never during release/device execution. */
    GENDIL_HOST_DEVICE GENDIL_INLINE
    constexpr void Validate() const
    {
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
        if constexpr ( has_complete_extents_v< Extents > )
        {
            if ( GetLogicalSize( extents ) == 0 ) { return; }
            GlobalIndex maximum_offset = 0;
            ConstexprLoop< Rank >( [&]( auto axis )
            {
                const auto count = GetExtentValue< axis >( extents ) - 1;
                GENDIL_ASSERT( strides[axis] == 0 || count <=
                    ( std::numeric_limits< GlobalIndex >::max() - maximum_offset ) /
                        strides[axis],
                    "Strided layout offset overflows GlobalIndex." );
                maximum_offset += count * strides[axis];
            } );
        }
#endif
    }
};

namespace detail {

    /** Compute prefix strides in one pass, checking each product in CPU debug. */
    template < size_t Rank, ExtentIndex IndexType >
    GENDIL_HOST_DEVICE
    constexpr auto FIFOStrides( const std::array< IndexType, Rank > & sizes )
    {
        std::array< GlobalIndex, Rank > strides{};
        GlobalIndex stride = 1;
        for ( size_t axis = 0; axis < Rank; ++axis )
        {
            const auto size = CheckedExtentCast< GlobalIndex >( sizes[axis] );
            strides[axis] = stride;
            GENDIL_ASSERT(
                stride == 0 || size <= std::numeric_limits< GlobalIndex >::max() / stride,
                "FIFO strided layout stride overflows GlobalIndex." );
            stride *= size;
        }
        return strides;
    }
}

/**
 * @brief Describe a tensor stored with its first coordinate varying fastest.
 *
 * For sizes {2,3}, entry (i,j) is at i + 2*j. Combine the result with a
 * container using MakeView, or use MakeFIFOView to perform both steps.
 * Sizes determine strides but are not retained, since indexing needs only
 * the spacing. Construct StridedLayout with Extents if shape queries are needed.
 *
 * Each stride is the product of preceding sizes; the axis-0 stride is one.
 * @pre Dimensions are nonnegative integral counts. Every intermediate product,
 * including the final multiplication, must fit in GlobalIndex even before a
 * later zero: {0,3} gives strides {1,0}, not an all-zero special case.
 * An empty dimension array produces a rank-zero layout.
 */
template < size_t Rank, detail::ExtentIndex IndexType >
GENDIL_HOST_DEVICE
constexpr auto MakeFIFOStridedLayout( std::array< IndexType, Rank > const & sizes )
{
    return StridedLayout< Rank >{ detail::FIFOStrides( sizes ), {} };
}

/** Tensor view over a raw-pointer buffer; stores spacing but not dimensions. */
template < Integer Rank, typename T = Real >
using StridedView = View< PointerContainer< T >, StridedLayout< Rank > >;
/** Tensor view over host/device storage; memory transfers remain explicit. */
template < Integer Rank, typename T = Real >
using HostDeviceStridedView = View< HostDevicePointer< T >, StridedLayout< Rank > >;

/**
 * View an existing buffer as a first-coordinate-fastest tensor, without copying it.
 * The buffer must outlive access through the view. Sizes determine spacing but
 * are not retained; MakeFIFOStridedLayout describes the size preconditions.
 */
template < typename T, typename IndexType, Integer dim >
GENDIL_HOST_DEVICE
auto MakeFIFOView(
    T* const data,
    std::array< IndexType, dim > const & sizes )
{
    return MakeView( data, MakeFIFOStridedLayout( sizes ) );
}

/**
 * View host/device storage as a first-coordinate-fastest tensor.
 * The handle is copied, not the values. Keep the selected buffer alive and
 * accessible; this factory performs no transfer or synchronization.
 */
template < typename T, typename IndexType, Integer dim >
GENDIL_HOST_DEVICE
auto MakeFIFOView( const HostDevicePointer< T > & data, std::array< IndexType, dim > const & sizes )
{
    return MakeView( data, MakeFIFOStridedLayout( sizes ) );
}

/** Raw-pointer FIFO factory accepting individual dimensions. */
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

/** Host/device FIFO factory accepting individual dimensions. */
template < typename T, typename... Args >
GENDIL_HOST_DEVICE
auto MakeFIFOView( const HostDevicePointer< T > & data, Args const & ... sizes )
{
    constexpr Integer rank = sizeof...(Args);
    return MakeFIFOView( data, std::array< GlobalIndex, rank>{
        detail::CheckedExtentCast< GlobalIndex >( sizes )... } );
}

/**
 * @brief Access a tensor whose entries are selected by an index map.
 *
 * Coordinates first identify an entry in FIFO order; the map then selects
 * its position in data: data[indirections[layout.Offset(indices...)]]. The child
 * container is copied according to its ownership semantics; the map is borrowed.
 * Both must remain valid in the execution memory space. No transfer or
 * synchronization is performed, and dimensions are not retained.
 */
template < typename T, typename... Args >
GENDIL_HOST_DEVICE
auto MakeIndirectedFIFOView( const T & data, const HostDevicePointer< const int > & indirections, Args const & ... sizes )
{
    constexpr Integer rank = sizeof...(Args);
    using Container = IndirectedContainer< T >;
    return MakeView( Container{ data, indirections }, MakeFIFOStridedLayout(
        std::array< GlobalIndex, rank>{
            detail::CheckedExtentCast< GlobalIndex >( sizes )... } ) );
}

namespace detail {

    /** Compute prefixes in permutation order, retaining strides in logical-axis order. */
    template < Integer Dim >
    GENDIL_HOST_DEVICE
    constexpr std::array< GlobalIndex, Dim > ComputedPermutedStrides(
        std::array< GlobalIndex, Dim > const & sizes,
        Permutation< Dim > const & permutations )
    {
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
        std::array< bool, Dim > seen{};
        for ( size_t axis = 0; axis < Dim; ++axis )
        {
            const auto selected = permutations( axis );
            GENDIL_ASSERT( selected > 0 && static_cast< size_t >( selected ) <= Dim,
                "Unsigned strided layout requires a positive axis permutation." );
            const auto selected_axis = static_cast< size_t >( selected - 1 );
            GENDIL_ASSERT( !seen[selected_axis],
                "Strided layout permutation repeats an axis." );
            seen[selected_axis] = true;
        }
#endif
        std::array< GlobalIndex, Dim > strides{};
        GlobalIndex current_stride = 1;
        for (GlobalIndex i = 0; i < Dim; i++)
        {
            for (GlobalIndex j = 0; j < Dim; j++)
            {
                if ( permutations( j ) == ( i+1 ) )
                {
                    strides[ j ] = current_stride;
                    GENDIL_ASSERT(
                        current_stride == 0 || sizes[j] <=
                            std::numeric_limits< GlobalIndex >::max() / current_stride,
                        "Permuted strided layout size overflows GlobalIndex." );
                    current_stride *= sizes[j];
                    break;
                }                
            }
        }
        return strides;
    }

    /** Pack selected computed strides into a constructor-argument tuple. */
    template < Integer dim, Integer... Is >
    GENDIL_HOST_DEVICE
    constexpr auto PermutationAndSizesToStridesAndSizes(
        std::array< GlobalIndex, dim > const & sizes,
        std::array< GlobalIndex, dim > const & permuted_strides,
        std::index_sequence< Is... > )
    {
        return std::make_tuple( permuted_strides[ Is ]... );
    }

    /** Compute permuted strides and return them as a tuple. */
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

/**
 * @brief Describe a tensor with a chosen fastest-to-slowest axis order.
 *
 * Use this when the first coordinate is not necessarily adjacent in storage.
 * The result changes how coordinates locate entries, not the order in which
 * callers provide coordinates, and does not move any data.
 *
 * permutation(j) specifies axis j's position in storage order: 1 means fastest.
 * Sizes remain in logical-axis order. For sizes {2,3,4}, permutation {3,1,2}
 * makes axis 1 fastest, then axis 2, then axis 0, giving strides {12,1,3}.
 * @pre Values 1 through dim occur exactly once; reversals are unsupported.
 * As for FIFO construction, all prefix products must fit, including for empty
 * shapes. Neither sizes nor the permutation are retained in the result.
 */
template < Integer dim >
GENDIL_HOST_DEVICE
constexpr auto MakeStridedLayout(
    std::array< GlobalIndex, dim > const & sizes,
    Permutation< dim > const & permutation )
{
    return StridedLayout< dim >{ detail::ComputedPermutedStrides( sizes, permutation ), {} };
}

/** View a buffer in the axis order described by MakeStridedLayout, without copying it. */
template < typename T, Integer dim >
GENDIL_HOST_DEVICE
auto MakeStridedView(
    T* const data,
    std::array< GlobalIndex, dim > const & sizes,
    Permutation< dim > const & permutation )
{
    return MakeView( data, MakeStridedLayout( sizes, permutation ) );
}

/** View host/device storage in the chosen axis order; returns a View, not just a layout. */
template < typename T, Integer dim >
GENDIL_HOST_DEVICE
constexpr auto MakeStridedLayout( HostDevicePointer< T > const & data, std::array< GlobalIndex, dim > const & sizes, Permutation< dim > const & permutation )
{
    return MakeView( data, MakeStridedLayout( sizes, permutation ) );
}

}
