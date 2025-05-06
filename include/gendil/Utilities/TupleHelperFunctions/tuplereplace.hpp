// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/IndexSequenceHelperFunctions/makeindexrange.hpp"
#include "getsubtuple.hpp"

namespace gendil {

namespace internal
{

template < typename NewType, typename... CurrentTypes >
constexpr auto tuple_replace_helper( std::tuple< CurrentTypes... > const & tuple,
                                     NewType const & new_value,
                                     std::index_sequence< 0 > )
{
   // constexpr size_t tuple_size = std::tuple_size_v< decltype( tuple ) >;
   constexpr size_t tuple_size = sizeof...(CurrentTypes);
   return   std::tuple_cat( std::make_tuple( new_value ), get_subtuple( tuple, make_index_range< 1, tuple_size >{} ) );
}

template < typename NewType, typename... CurrentTypes >
constexpr auto tuple_replace_helper( std::tuple< CurrentTypes... > const & tuple,
                                     NewType const & new_value,
                                     std::index_sequence< sizeof...(CurrentTypes) > )
{
   constexpr size_t tuple_size = std::tuple_size_v< decltype( tuple ) >;
   return   std::tuple_cat( get_subtuple( tuple, make_index_range< 0, tuple_size-1 >{} ), std::make_tuple( new_value ) );
}

template < size_t index, typename NewType, typename... CurrentTypes >
constexpr auto tuple_replace_helper( std::tuple< CurrentTypes... > const & tuple,
                                     NewType const & new_value,
                                     std::index_sequence< index > )
{
   constexpr size_t tuple_size = sizeof...( CurrentTypes );
   static_assert( index > 0, "This function shouldn't be called with index == 0." );
   static_assert( index < tuple_size, "Trying to insert beyond the end of the tuple." );
   return   std::tuple_cat(
               get_subtuple( tuple, make_index_range< 0, index >{} ),
               std::make_tuple( new_value ),
               get_subtuple( tuple, make_index_range< index+1, tuple_size >{} )
            );
}

}

template < size_t index, typename NewType, typename... CurrentTypes >
constexpr auto tuple_replace( std::tuple< CurrentTypes... > const & tuple, NewType const & new_value )
{
   return internal::tuple_replace_helper( tuple, new_value, std::index_sequence< index >{} );
}

}
