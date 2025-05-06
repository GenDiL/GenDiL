// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <utility>

#include "subsequence.hpp"

namespace gendil
{

/// @brief alias for the first N elements of index sequence Sequence
/// @tparam Sequence an std::index_sequence
/// @tparam N number of elements
template < size_t N, typename Sequence >
using SequenceHead = subsequence_t< Sequence, std::make_index_sequence< N > >;

template < size_t N, typename Sequence >
using SequenceTail = subsequence_t< Sequence, std::make_index_sequence< N >, Sequence::size() - N >;

/// @brief Splits sequence into `Head`: the first N elements, and `Tail`: the last size - N elements.
/// @tparam Sequence an index sequence
/// @tparam N number of elements at the start of the sequence to grab
template < size_t N, typename Sequence >
struct HeadTailSplit
{
   using Head = SequenceHead< N, Sequence >;
   using Tail = SequenceTail< Sequence::size() - N, Sequence >;
};

/// @brief Splits sequence into `Head`: first size - N elements, and `Tail`: the last N, elements.
/// @tparam Sequence  an index sequence
/// @tparam N number of elements at the end of the sequence to grab
template < size_t N, typename Sequence >
using TailHeadSplit = HeadTailSplit< Sequence::size() - N, Sequence >;

} // namespace gendil
