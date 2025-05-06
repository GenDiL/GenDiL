// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/reverseindexsequence.hpp"

namespace gendil {

/**
 * @brief Nested for loop abstraction with last indice running faster.
 * 
 * @tparam Lambda The type of the body of the for nested loops.
 * @tparam I The size of the least nested loop.
 * @tparam Args The types of the nested loops indices.
 * @param lambda The body of the nested loops.
 * @param args The indices of the nest loops.
 * @return GENDIL_HOST_DEVICE 
 */
template < int last_dim, typename Lambda, typename... Args >
GENDIL_HOST_DEVICE
void ReverseLoop( Lambda && lambda, Args && ... args )
{
   for (GlobalIndex ind = 0; ind < last_dim; ind++)
   {
      lambda( args..., ind );
   }
}

template < int first_dim, int second_dim, int... rest_dims, typename Lambda, typename... Args >
GENDIL_HOST_DEVICE
void ReverseLoop( Lambda && lambda, Args && ... args )
{
   for (GlobalIndex ind = 0; ind < first_dim; ind++)
   {
      ReverseLoop< second_dim, rest_dims... >( std::forward< Lambda >( lambda ), args..., ind );
   }
}

/**
 * @brief Nested for loop abstraction with first indice running faster.
 * 
 * @tparam Lambda The type of the body of the for nested loops.
 * @tparam I The size of the outer most loop.
 * @tparam Is The sizes of the nested loop (or nothing).
 * @param lambda The body of the nested loops.
 * @param args The indices of the nest loops.
 * @return GENDIL_HOST_DEVICE 
*/
template < typename Lambda, size_t I, size_t... Is, typename... Args >
GENDIL_HOST_DEVICE
void ForwardLoop( Lambda && lambda, std::index_sequence< I, Is... >, Args && ... args)
{
   for (GlobalIndex ind = 0; ind < I; ind++)
   {
      if constexpr (sizeof...(Is) > 0)
      {
         ForwardLoop( std::forward< Lambda >( lambda ), std::index_sequence< Is... >{}, ind, std::forward< Args >( args )... );
      }
      else
      {
         lambda( ind, std::forward< Args >( args )... );
      }
   }
}

template < size_t... dims, typename Lambda, typename... Args >
GENDIL_HOST_DEVICE
void Loop( Lambda && lambda, Args && ... args )
{
   using seq = ReverseIndexSequence< dims... >;
   ForwardLoop( std::forward< Lambda >( lambda ), seq{}, std::forward< Args >( args )... );
   // ReverseLoop<dims...>( lambda, args... );
}

template < typename Dims, typename Lambda, typename... Args >
GENDIL_HOST_DEVICE
void Loop( Lambda && lambda, Args && ... args )
{
   using seq = typename ReverseIndexSequence_t< Dims >::type;
   ForwardLoop( std::forward< Lambda >( lambda ), seq{}, std::forward< Args >( args )... );
}

}
