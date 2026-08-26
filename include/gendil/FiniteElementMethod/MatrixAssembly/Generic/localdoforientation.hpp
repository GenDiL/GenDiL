// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofdescriptor.hpp"
#include "gendil/Meshes/Connectivities/Orientations/referencetonativeindex.hpp"
#include "gendil/Utilities/toarray.hpp"

namespace gendil {

template <
   typename FESpace,
   typename Descriptor,
   typename Orientation >
GENDIL_HOST_DEVICE
auto OrientReferenceDofToNative(
   const FESpace &,
   const Descriptor & dof,
   const Orientation & orientation )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      finite_element_space_shape_functions_t< Space >;
   using DofDescriptor = std::remove_cvref_t< Descriptor >;
   using ComponentDofShape =
      component_dof_shape_t< ShapeFunctions, DofDescriptor::component_id >;

   static_assert(
      orientation_dimension_v< Orientation > ==
         ComponentDofShape::size(),
      "The local DoF shape and orientation dimensions must match." );

   const auto native_indices = ReferenceToNativeIndex(
      dof.indices,
      to_array( ComponentDofShape{} ),
      orientation );

   return MakeLocalDofDescriptorFromFullIndices( dof, native_indices );
}

} // namespace gendil
