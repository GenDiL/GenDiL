// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "cat.hpp"

namespace gendil 
{

template < size_t val, size_t N >
struct repeat
{
   using type = cat_t< typename repeat< val, N-1 >::type, std::index_sequence< val > >;
};

template < size_t val >
struct repeat< val, 0 >
{
   using type = std::index_sequence<>;
};

template < size_t val, size_t N >
using repeat_t = typename repeat< val, N >::type;

}