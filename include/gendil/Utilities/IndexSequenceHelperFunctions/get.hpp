// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <utility>

namespace gendil {

template < size_t Index, typename Sequence >
struct SequenceGet;

template < size_t Index, size_t... I >
struct SequenceGet< Index, std::index_sequence< I... > >
{
    static constexpr size_t value = std::tuple_element_t< Index, std::tuple< std::integral_constant< size_t, I >... > >::value;
};

/**
 * @brief returns the value of the `Sequence` at position `Index`.
*/
template < size_t Index, typename Sequence >
constexpr size_t seq_get_v = SequenceGet< Index, Sequence >::value;

template < size_t Index, size_t... I >
struct VariadicSequenceGet
{
    static constexpr size_t value = seq_get_v< Index, std::index_sequence< I... > >;
};

/**
 * @brief returns the value of the variadic pack `I...` at position `Index`.
*/
template < size_t Index, size_t... I >
constexpr size_t vseq_get_v = VariadicSequenceGet< Index, I... >::value;

}
