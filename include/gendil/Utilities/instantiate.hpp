// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementorders.hpp"

namespace gendil {

template < template <Integer...> typename T, typename IntList >
struct instantiate;

template < template <Integer...> typename T, Integer... Orders >
struct instantiate< T, FiniteElementOrders< Orders... > >
{
    using type = T<Orders...>;
};

/**
 * @brief Instantiate a template with a sequence of orders.
 * 
 * @tparam T The template to instantiate.
 * @tparam IntList The values to instantiate the template.
 */
template < template <Integer...> typename T, typename IntList >
using instantiate_t = typename instantiate< T, IntList >::type;

}
