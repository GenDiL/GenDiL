// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

/**
 * @brief Concatenate two std::index_sequence.
 * 
 * @tparam Is The indices from the first index sequence.
 * @tparam Js The indices from the second index sequence.
 * @return constexpr auto The resulting index sequence.
 */
template < size_t... Is, size_t... Js >
constexpr auto cat( std::index_sequence< Is... >, std::index_sequence< Js... > )
{
    return std::index_sequence< Is..., Js... >{};
}

template < typename... Sequences >
struct CatType;

template < size_t... I >
struct CatType< std::index_sequence<I...> >
{
    using type = std::index_sequence<I...>;
};

template < size_t... I, size_t... J, typename... Sequences >
struct CatType< std::index_sequence< I... >, std::index_sequence< J... >, Sequences... >
{
    using type = typename CatType< std::index_sequence< I..., J... >, Sequences... >::type;
};

/// @brief concatenate two std::index_sequence
template < typename... Sequences >
using cat_t = typename CatType< Sequences... >::type;

}
