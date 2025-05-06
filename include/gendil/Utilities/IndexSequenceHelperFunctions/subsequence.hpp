// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <utility>

#include "get.hpp"

namespace gendil
{

template < typename Sequence, typename SubsequenceIndices, size_t Offset = 0 >
struct SubSequence;

/**
 * @brief computes an index_sequence whose elements are those of `Sequence`
 * selected at `SubsequenceIndices` offset by `Offset`.
 * 
 * @details example:
 * ```
 * using sequence = std::index_sequence< 0, 1, 0, 2, 0, 3 >;
 * using indices = std::make_index_sequence< 3 >;
 * 
 * using subsequence = subsequence_t< sequence, indices, 1 >;
 * static_assert( std::is_same_v< subsequence, std::index_sequence< 1, 2, 3 > > );
 * ```
*/
template < typename Sequence, typename SubsequenceIndices, size_t Offset = 0 >
using subsequence_t = typename SubSequence< Sequence, SubsequenceIndices, Offset >::type;

template < typename Sequence, size_t Offset, size_t... I >
struct SubSequence< Sequence, std::index_sequence<I...>, Offset >
{
   using type = std::index_sequence< seq_get_v<Offset + I, Sequence >... >;
};

} // namespace gendil
