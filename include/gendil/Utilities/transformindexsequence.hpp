// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil
{

template < template <size_t> typename Transformer, typename IndexSequence >
struct TransformIndexSequence;

template < template <size_t> typename Transformer, size_t ... Is >
struct TransformIndexSequence< Transformer, std::index_sequence< Is ... > >
{
   using type = typename std::index_sequence< Transformer< Is >::value ... >;
};

/**
 * @brief an index_sequence whose elements are those of IndexSequence
 * transformed by Transformer.
 *
 * @tparam Transformer a template type such that Transformer< I > has a static
 * constexpr member `value` which is the value of the transformation of `I`.
 * @tparam IndexSequence the std::index_sequence to transform
*/
template < template < size_t > typename Transformer, typename IndexSequence >
using transform_index_sequence_t = typename TransformIndexSequence< Transformer, IndexSequence >::type;

} // namespace gendil
