// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofdescriptor.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/toarray.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"

namespace gendil {

template <
   typename DofShape,
   Integer Dim >
GENDIL_HOST_DEVICE
auto ReferenceToNativeIndexFromOrientedLayout(
   const std::array< GlobalIndex, Dim > & reference_indices,
   const std::array< size_t, Dim > & dof_sizes,
   const Permutation< Dim > & orientation )
{
   // Vector DoF reading writes native indices through MakeOrientedLayout and
   // reads them back through FIFO/reference order. For vector sparse insertion,
   // invert that exact storage-layout mapping instead of using the scalar
   // axis formula. Anisotropic vector component shapes are currently validated
   // for Cartesian/layout-level tests only, not claimed for unstructured meshes.
   const auto oriented_layout =
      MakeOrientedLayout( dof_sizes, orientation );
   const GlobalIndex reference_offset =
      FlattenMultiIndex< DofShape >( reference_indices );

   bool found = false;
   std::array< GlobalIndex, Dim > native_indices{};

   UnitLoop< DofShape >( [&] ( auto... k )
   {
      if ( !found &&
           oriented_layout.Offset( k... ) == reference_offset )
      {
         native_indices =
            std::array< GlobalIndex, Dim >{
               static_cast< GlobalIndex >( k )... };
         found = true;
      }
   });

   GENDIL_VERIFY(
      found,
      "Unable to map reference DoF index to native index through oriented layout." );

   return native_indices;
}

template <
   typename FESpace,
   typename Descriptor,
   typename Orientation >
GENDIL_HOST_DEVICE
auto OrientReferenceDofToNative(
   const FESpace &,
   const Descriptor & dof,
   const Orientation & orientation_ )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions = typename Space::finite_element_type::shape_functions;
   using DofDescriptor = std::remove_cvref_t< Descriptor >;
   using ComponentDofShape =
      component_dof_shape_t< ShapeFunctions, DofDescriptor::component_id >;

   Permutation< Space::Dim > orientation = orientation_;
   const auto dof_sizes = to_array( ComponentDofShape{} );

   auto native_indices = [&]
   {
      if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
      {
         // Match the vector DoF read path. Scalar keeps the previously
         // validated ReferenceToNativeIndex path below.
         return ReferenceToNativeIndexFromOrientedLayout< ComponentDofShape >(
            dof.indices,
            dof_sizes,
            orientation );
      }
      else
      {
         return ReferenceToNativeIndex(
            dof.indices,
            dof_sizes,
            orientation );
      }
   }();

   return MakeLocalDofDescriptorFromFullIndices( dof, native_indices );
}

} // namespace gendil
