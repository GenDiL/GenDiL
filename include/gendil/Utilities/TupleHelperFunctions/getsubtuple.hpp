// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

template < typename Tuple, size_t... Is >
constexpr auto get_subtuple( Tuple const & tuple, std::index_sequence<Is...> )
{
    return std::make_tuple( std::get< Is >( tuple )... );
}

}
