// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>

#include "types.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

namespace gendil
{

namespace details
{

template < Integer Dim, Integer Dims >
GENDIL_HOST_DEVICE
constexpr void ComputeStructuredSubIndices( std::array< GlobalIndex, Dims > & sub_index, GlobalIndex & index, std::array< GlobalIndex, Dims > const & sizes )
{
    if constexpr ( Dim == 2 )
    {
        sub_index[ 1 ] = index / sizes[ 0 ];
        sub_index[ 0 ] = index - sizes[ 0 ] * sub_index[ 1 ];
    }
    else
    {
        // We could pass n to this function and divide by sizes[ Dim-1 ] to
        // compute the strides, but division is x10 more expensive on GPU. For
        // most meshes, this is more efficient.
        const GlobalIndex n = Product( sizes, std::make_index_sequence< Dim-1 >{} );
        sub_index[ Dim-1 ] = index / n;

        index -= n * sub_index[ Dim-1 ]; // index = index % n

        ComputeStructuredSubIndices< Dim-1 >( sub_index, index, sizes );
    }
}

template < Integer Dim, Integer Back, Integer Dims >
GENDIL_HOST_DEVICE
constexpr GlobalIndex ComputeStructuredSubIndex( GlobalIndex & index, std::array< GlobalIndex, Dims > const & sizes )
{
    static_assert( Dim <= Back );
    static_assert( Back <= Dims );
    
    if constexpr ( Dim == 0 ) // shortcut for innner-most dimension
    {
        return index % sizes[ 0 ];
    }
    else if constexpr ( Dim == 1 and Back == 1 ) // shortcut second dimension
    {
        return index / sizes[ 0 ];
    }
    else
    {
        const GlobalIndex n = Product( sizes, std::make_index_sequence< Back-1 >{} );

        if constexpr ( Dim+1 == Back )
        {
            return index / n;
        }
        else if constexpr ( Dim == Back )
        {
            return index % n;
        }
        else
        {
            index %= n;
            return ComputeStructuredSubIndex< Dim, Back-1, Dims >( index, sizes );
        }
    }
}

} // namespace details

/**
 * @brief computes all structured sub-indices for caresian products.
 * 
 * @tparam Dim the number of dimensions in the cartesian product.
 * @param index the linear index.
 * @param sizes the sizes of the cartesian product structure along each dimension.
*/
template < Integer Dims >
GENDIL_HOST_DEVICE
constexpr std::array< GlobalIndex, Dims > GetStructuredSubIndices( GlobalIndex index, std::array< GlobalIndex, Dims > const & sizes )
{
    static_assert( Dims >= 2, "GetStructuredSubIndices is not defined for Dim < 2." );
    
    std::array< GlobalIndex, Dims > sub_index;
    details::ComputeStructuredSubIndices< Dims >( sub_index, index, sizes );

    return sub_index;
}

/**
 * @brief Computes the sub-index for dimension Dim in cartesian product structure with Dims dimensions.
*/
template < Integer Dim, Integer Dims >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetStructuredSubIndex( GlobalIndex index, std::array< GlobalIndex, Dims > const & sizes )
{
    static_assert( Dims >= 2 );
    static_assert( Dim < Dims );

    return details::ComputeStructuredSubIndex< Dim, Dims, Dims >( index, sizes );
}

} // namespace gendil
