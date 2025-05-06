// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

template < Integer index, Integer first_order, Integer... rest_orders >
struct get_order
{
   static constexpr Integer value = get_order< index-1, rest_orders... >::value;
};

template < Integer first_order, Integer... rest_orders >
struct get_order< 0, first_order, rest_orders... >
{
   static constexpr Integer value = first_order;
};

}
