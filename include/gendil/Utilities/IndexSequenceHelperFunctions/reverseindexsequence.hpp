// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

template < typename IndexSequence, typename ReversedIndexSequence = std::index_sequence<> >
struct ReverseIndexSequence_t;

template < size_t I, size_t... Is, size_t... Js >
struct ReverseIndexSequence_t< std::index_sequence< I, Is... >, std::index_sequence< Js... > >
{
   using type = typename ReverseIndexSequence_t< std::index_sequence< Is... >, std::index_sequence< I, Js... > >::type;
};

template < size_t... Js >
struct ReverseIndexSequence_t< std::index_sequence<>, std::index_sequence< Js... > >
{
   using type = std::index_sequence< Js... >;
};

template < size_t... Is >
using ReverseIndexSequence = typename ReverseIndexSequence_t< std::index_sequence< Is... > >::type;

}
