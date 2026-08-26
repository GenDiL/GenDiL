// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Structured contiguous and indirect tensor-view construction helpers.
 */

#include "gendil/FiniteElementMethod/ShapeFunctions/finiteelementorders.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

#include <utility>

namespace gendil {

/** @brief View finite-element data using explicit polynomial orders. */
template <
   typename FiniteElementSpace,
   Integer ... Orders,
   typename T >
auto MakeTensor(
   FiniteElementSpace const & finite_element_space,
   T * data,
   FiniteElementOrders< Orders... > )
{
   const GlobalIndex num_elem = finite_element_space.GetNumberOfFiniteElements();
   return MakeFIFOView( data, ((GlobalIndex)Orders + 1)..., num_elem );
}

/** @brief View finite-element data using the space's scalar DoF orders. */
template < typename FiniteElementSpace, typename T >
auto MakeTensor(
   FiniteElementSpace const & finite_element_space,
   T * data )
{
   using orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   return MakeTensor( finite_element_space, data, orders{} );
}

/** @brief View contiguous data with an explicit static local DoF shape. */
template < typename T, size_t ... dof_shapes >
auto MakeTensor(
   T * data,
   GlobalIndex num_elements,
   std::index_sequence< dof_shapes... > )
{
   return MakeFIFOView( data, (GlobalIndex)dof_shapes..., num_elements );
}

/** @brief View indirectly addressed data with an explicit static DoF shape. */
template < typename T, size_t ... dof_shapes >
auto MakeIndirectedTensor(
   T * data,
   const HostDevicePointer< const int > & indices,
   GlobalIndex num_elements,
   std::index_sequence< dof_shapes... > )
{
   return MakeIndirectedFIFOView(
      data,
      indices,
      static_cast< GlobalIndex >( dof_shapes )...,
      num_elements );
}

} // namespace gendil
