// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "find.hpp"
#include "replace.hpp"

namespace gendil
{
   namespace details
   {
      template < bool base_case, size_t iter, typename Permutation >
      struct InversePermutation;

      template < size_t iter, typename Permutation >
      struct InversePermutation< true, iter, Permutation >
      {
         static constexpr size_t position = sequence_find< iter, Permutation >;
         using type = replace_t< iter, position, Permutation >;
      };

      template < size_t iter, typename Permutation >
      struct InversePermutation< false, iter, Permutation >
      {
         static constexpr size_t position = sequence_find< iter, Permutation >;
         using type = replace_t < iter, position, typename InversePermutation< (iter+1 >= Permutation::size()), iter+1, Permutation >::type >;
      };
   } // namespace details

   /**
    * @brief computes the inverse permutation of `Permutation` where
    * `Permuatation` is an `std::index_sequence` which is some permutation of
    * `std::make_index_sequence< N >` (`N = Permutation::size()`).
    * 
    * @details example:
    * ```
    * using perm = std::index_sequence< 2, 0, 1 >;
    * using inv_perm = inverse_permutation< perm >;
    * 
    * static_assert( std::is_same_v< inv_perm, std::index_sequence< 1, 2, 0 > >);
    * ```
   */
   template < typename Permutation >
   using inverse_permutation = typename details::InversePermutation< (1 >= Permutation::size()),  0, Permutation >::type;

} // namespace gendil
