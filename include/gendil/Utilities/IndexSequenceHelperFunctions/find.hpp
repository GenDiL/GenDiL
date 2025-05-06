// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil
{
   namespace details
   {
      template < size_t value, size_t iter, size_t... sequence >
      struct SequenceFind;

      template < size_t value, size_t iter, size_t head, size_t... tail >
      struct SequenceFind< value, iter, head, tail... >
      {
         static constexpr size_t position = (value == head) ? iter : SequenceFind<value, iter+1, tail...>::position;
      };

      template < size_t value, size_t iter >
      struct SequenceFind< value, iter >
      {
         static constexpr size_t position = iter;
      };

      template < size_t value, size_t iter, typename IndexSequence >
      struct IndexSequenceFind;

      template < size_t value, size_t iter, size_t... sequence >
      struct IndexSequenceFind< value, iter, std::index_sequence< sequence... > >
      {
         static constexpr size_t position = SequenceFind< value, iter, sequence... >::position;
      };
   } // namespace details
   
   /**
    * @brief computes the position in the `IndexSequence` where `value` is found.
    * 
    * @details example:
    * ```
    * using A = std::index_sequence< 0, 100, 200, 300 >;
    * 
    * constexpr size_t position = sequence_find< 200, A >;
    * static_assert( position == 2 );
    * ```
   */
   template < size_t value, typename IndexSequence >
   inline constexpr size_t sequence_find = details::IndexSequenceFind< value, 0, IndexSequence >::position;

} // namespace gendil
