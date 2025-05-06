// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "getsubtuple.hpp"

namespace gendil
{
   namespace details
   {
      template < size_t offset, typename Sequence >
      struct OffsetSequence;

      template < size_t offset, size_t... I >
      struct OffsetSequence< offset, std::index_sequence<I...> >
      {
         using type = std::index_sequence< (offset + I)... >;
      };

      template < size_t offset, size_t N >
      using make_offset_sequence = typename OffsetSequence< offset, std::make_index_sequence<N> >::type;
   } // namespace details
   

   /// @brief returns the first `N` elements of the tuple as another tuple.
   template < size_t N, typename Tuple >
   constexpr auto tuple_head( const Tuple & tup )
   {
      return get_subtuple( tup, std::make_index_sequence<N>{} );
   }

   /// @brief returns the last `N` elements of the tuple as another tuple.
   template < size_t N, typename Tuple >
   constexpr auto tuple_tail( const Tuple & tup )
   {
      constexpr size_t offset = std::tuple_size_v<Tuple> - N;
      return get_subtuple( tup, details::make_offset_sequence< offset, N >{} );
   }
}