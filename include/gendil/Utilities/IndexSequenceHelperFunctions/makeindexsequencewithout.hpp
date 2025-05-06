// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

namespace internal {

template < size_t begin, size_t end, size_t without, size_t... indices >
struct IndexSequenceWithoutHelper
   : public IndexSequenceWithoutHelper< begin+1, end, without, indices..., begin >
{};

template < size_t begin, size_t end, size_t... indices >
struct IndexSequenceWithoutHelper< begin, end, begin, indices...>
   : IndexSequenceWithoutHelper< begin+1, end, begin, indices...>
{};

template < size_t begin, size_t without, size_t... indices >
struct IndexSequenceWithoutHelper< begin, begin, without, indices...>
{
   using type = std::index_sequence< indices... >;
};

}

/**
 * @brief Helper type to create an index sequence without a provided value.
 * 
 * @tparam end The end of the index sequence (excluded)
 * @tparam without The value to be excluded from the sequence.
 */
template < size_t end, size_t without >
using make_index_sequence_without = typename internal::IndexSequenceWithoutHelper< 0, end, without >::type;

}
