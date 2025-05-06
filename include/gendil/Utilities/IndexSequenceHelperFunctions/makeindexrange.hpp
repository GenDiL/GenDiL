// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

template < size_t begin, size_t end, size_t... indices >
struct IndexRangeHelper : public IndexRangeHelper< begin+1, end, indices..., begin >
{};

template < size_t begin, size_t... indices >
struct IndexRangeHelper< begin, begin, indices...>
{
   using type = std::index_sequence< indices... >;
};

/**
 * @brief Helper type to create ranges of values instead of sequences starting from 0.
 * 
 * @tparam begin The begining of the sequence (included)
 * @tparam end The end of the sequence (excluded)
 */
template < size_t begin, size_t end >
using make_index_range = typename IndexRangeHelper< begin, end >::type;

}
