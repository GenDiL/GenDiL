#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

template < typename T >
struct get_rank;

template < typename Container, typename Layout >
struct View;

template < typename Container, typename Layout >
struct get_rank< View< Container, Layout > >
{
   static constexpr Integer value = View< Container, Layout >::rank;
};

template < typename FiniteElementSpace >
struct ElementDoF;

template < typename FiniteElementSpace >
struct get_rank< ElementDoF< FiniteElementSpace > >
{
   static constexpr Integer value = ElementDoF< FiniteElementSpace >::Dim;
};

template< typename T, typename... DimensionInfo >
struct RecursiveArray;

template< typename T, typename... DimensionInfo >
struct get_rank< RecursiveArray< T, DimensionInfo... > >
{
   static constexpr Integer value = RecursiveArray< T, DimensionInfo... >::rank;
};

template< typename Sizes, typename KernelContext, typename Container >
struct ThreadedView;

template< typename Sizes, typename KernelContext, typename Container >
struct get_rank< ThreadedView< Sizes, KernelContext, Container > >
{
   // !FIXME Is this really what we want?
   static constexpr Integer value = get_rank< Container >::value;
};

template < typename T >
static constexpr Integer get_rank_v = get_rank< std::decay_t< T > >::value;

}
