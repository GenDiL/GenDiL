// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/get.hpp"

namespace gendil {

// TODO: Rename to TensorFiniteElementOrders? -> Remove?
/**
 * @brief Simple structure representing a list of polynomial orders for each dimension.
 * 
 * @tparam Orders The list of polynomial orders.
 */
template < Integer... Orders >
struct FiniteElementOrders
{
   static constexpr Integer dim = sizeof...( Orders );
};

template < size_t Dim, typename Orders >
struct GetNumDofs
{
   static constexpr size_t value = seq_get_v< Dim, Orders >;
};

template < size_t Dim, size_t ... Orders >
struct GetNumDofs<Dim, FiniteElementOrders< Orders... > >
{
   static constexpr size_t value = 1 + GetNumDofs<Dim, std::index_sequence<Orders...> >::value;
};

template < typename FEOrders >
struct OrdersToNumDofs;

template < Integer... Orders >
struct OrdersToNumDofs< FiniteElementOrders< Orders... > >
{
   using num_dofs = std::index_sequence< (Orders+1)... >;
};

template < typename FEOrders >
using orders_to_num_dofs = typename OrdersToNumDofs< FEOrders >::num_dofs;

}
