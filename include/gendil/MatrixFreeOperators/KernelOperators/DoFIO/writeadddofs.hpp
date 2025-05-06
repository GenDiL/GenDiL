// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/LoopHelpers/dofloop.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

/**
 * @brief Write and add element local degrees-of-freedom to a global tensor of degrees-of-freedom.
 * 
 * @tparam FiniteElementSpace The type of finite element space.
 * @tparam Dim The dimension of the finite element space.
 * @param element_index The index of the finite element.
 * @param local_dofs The container containing element local degrees-of-freedom.
 * @param global_dofs The tensor containing all the degrees-of-freedom.
 */
template < typename FiniteElementSpace,
           Integer Dim >
GENDIL_HOST_DEVICE
void WriteAddDofs(
   const GlobalIndex & element_index,
   const ElementDoF< FiniteElementSpace > & local_dofs,
   StridedView< Dim, Real > & global_dofs )
{
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in WriteAddDofs."
   );
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         AtomicAdd( global_dofs( indices..., element_index ), local_dofs( indices... ) );
      }
   );
}

}
