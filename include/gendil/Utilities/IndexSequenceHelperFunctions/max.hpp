// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <utility>
#include "gendil/Utilities/MathHelperFunctions/max.hpp"

namespace gendil {

template < typename LHSSequence, typename RHSSequence >
struct MaxSequence;

template < size_t... I, size_t... J >
struct MaxSequence< std::index_sequence< I... >, std::index_sequence< J... > >
{
    using type = std::index_sequence< Max(I,J)... >;
};

/**
 * @brief returns the sequence of the `Sequence` at position `Index`.
*/
template < typename LHSSequence, typename RHSSequence >
using max_sequence_t = typename MaxSequence< LHSSequence, RHSSequence >::type;

}