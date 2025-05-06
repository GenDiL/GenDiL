// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <utility>

#include "get.hpp"

namespace gendil
{

namespace details
{

template < size_t Index, size_t Value, typename Sequence, typename I >
struct ReplaceHelper;

template < size_t Index, size_t Value, typename Sequence, size_t... I >
struct ReplaceHelper< Index, Value, Sequence, std::index_sequence<I...> >
{
   using result = std::index_sequence< ((I == Index) ? Value : seq_get_v< I, Sequence >)... >;
};

} // namespace details

/**
 * @brief computes an `index_sequence` whose elements are those of `Sequence`
 * except at position `Index` where instead the sequence has value `Value`.
 * 
 * @details example:
 * ```
 * using A = std::index_sequence< 0, 1, 2, 100, 4 >;
 * 
 * using B = replace_t< 3, 3, A >;
 * static_assert( std::is_same_v< B, std::index_sequence< 0, 1, 2, 3, 4 > >);
 * ```
*/
template < size_t Index, size_t Value, typename Sequence >
using replace_t = typename details::ReplaceHelper< Index, Value, Sequence, std::make_index_sequence< Sequence::size() > >::result;

namespace details
{

template < typename A, typename B, typename I >
struct ReplaceSubSequenceHelper;

template < typename A, typename B >
struct ReplaceSubSequenceHelper< A, B, std::index_sequence<> >
{
   using result = A;
};

template < typename A, typename B, size_t I, size_t... Is >
struct ReplaceSubSequenceHelper< A, B, std::index_sequence<I, Is...> >
{
   using intermediate_result = replace_t< I, seq_get_v< I, B >, A >;
   using result = typename ReplaceSubSequenceHelper< intermediate_result, B, std::index_sequence<Is...> >::result;
};

} // namespace details

/**
 * @brief computes an `index_sequence` whose elements are those of `SequenceA`
 * except at the positions specified by `Indicator` where instead the sequence
 * has the values of `SequenceB`
 * 
 * @details example:
 * ```
 * using A = std::index_sequence< 100, 200, 3 >;
 * 
 * using B = std::index_sequence < 1, 2 >;
 * using Indicator = std::make_index_sequence< 2 >;
 * 
 * using C = replace_subsequence_t< A, B, Indicator >;
 * static_assert( std::is_same_v< C, std::index_sequence< 1, 2, 3 > >);
 * ```
*/
template < typename SequenceA, typename SequenceB, typename Indicator >
using replace_subsequence_t = typename details::ReplaceSubSequenceHelper< SequenceA, SequenceB, Indicator >::result;

} // namespace gendil
