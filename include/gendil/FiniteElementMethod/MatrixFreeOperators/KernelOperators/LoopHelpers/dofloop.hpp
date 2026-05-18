// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

/**
 * @brief A loop abstraction allowing to loop on all the local degrees-of-freedom indices.
 * 
 * @tparam FiniteElementSpace The finite element space type.
 * @tparam Lambda The type of the lambda function.
 * @tparam Args The list of arguments for the lambda function.
 * @param lambda The function to be called with each degree-of-freedom index.
 * @param args The arguments for the lambda function.
 */
template < typename FiniteElementSpace, typename Lambda, typename... Args >
GENDIL_HOST_DEVICE
void DofLoop( Lambda && lambda, Args && ... args );

/**
 * @brief A loop abstraction allowing to loop on all the local degrees-of-freedom indices.
 * 
 * @tparam Lambda The type of the lambda function.
 * @tparam Args The list of arguments for the lambda function.
 * @tparam Orders The finite element orders in each dimension.
 * @param lambda The function to be called with each degree-of-freedom index.
 * @param args The arguments for the lambda function.
 */
template < typename Lambda, typename... Args, Integer... Orders  >
GENDIL_HOST_DEVICE
void DofLoop( FiniteElementOrders<Orders...>, Lambda && lambda, Args && ... args )
{
   Loop< (Orders + 1)... >( std::forward< Lambda >( lambda ), std::forward< Args >( args ) ... );
}

template < typename FiniteElementSpace, typename Lambda, typename... Args >
GENDIL_HOST_DEVICE
void DofLoop( Lambda && lambda, Args && ... args )
{
   // Assumes tensor finite element.
   using orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   DofLoop( orders{}, std::forward< Lambda >( lambda ), std::forward< Args >( args ) ... );
}

template < typename ThreadLayout, typename Lambda, Integer... Orders >
GENDIL_HOST_DEVICE
void DofLoop( FiniteElementOrders< Orders... >, const ThreadLayout & thread, Lambda && lambda)
{
   ThreadLoop< (Orders+1) ... >( thread, std::forward< Lambda >( lambda ));
}

template < typename FiniteElementSpace, typename ThreadLayout, typename Lambda >
GENDIL_HOST_DEVICE
void DofLoop( const ThreadLayout & thread, Lambda && lambda )
{
   using orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   DofLoop( orders{}, thread, std::forward< Lambda >( lambda ));
}

}
