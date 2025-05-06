// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

namespace gendil {

/**
 * @brief Get the quadrature weight at a given quadrature point.
 * 
 * @tparam Dim The dimension of the space.
 * @tparam DofToQuads 1D DofToQuad objects for each dimension.
 * @param index The index of the quadrature point.
 * @param quad_data The DofToQuad objects containing 1D data for each dimension.
 * @return The quadrature weight.
 * 
 * @note Assumes tensor finite element with tensor integration rule.
 */
template < Integer Dim, typename... DofToQuads >
GENDIL_HOST_DEVICE
Real GetWeight( const TensorIndex< Dim > & index,
                const std::tuple< DofToQuads... > & quad_data );

template < Integer Dim,
           typename... DofToQuads,
           size_t... Is >
GENDIL_HOST_DEVICE
Real GetWeight( const TensorIndex< Dim > & index,
                const std::tuple< DofToQuads... > & quad_data,
                std::index_sequence< Is... > )
{
   return Product( std::get< Is >( quad_data ).weights( index[ Is ] )... );
}

template < Integer Dim, typename... DofToQuads >
GENDIL_HOST_DEVICE
Real GetWeight( const TensorIndex< Dim > & index,
                const std::tuple< DofToQuads... > & quad_data )
{
   return GetWeight( index,
                     quad_data,
                     std::make_index_sequence< Dim >{} );
}

}
