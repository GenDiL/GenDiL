// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"

namespace gendil {

/** @brief Default no-op hook for representation-specific validation. */
template < ElementDoFRestriction Restriction >
void ValidateRestrictionRepresentation( const Restriction & )
{ }

/**
 * @brief Validate the representation-independent invariants of a completed
 * restriction, followed by its ADL-selected representation-specific checks.
 */
template < ElementDoFRestriction Restriction >
void ValidateElementDoFRestriction( const Restriction & restriction )
{
   GENDIL_VERIFY(
      GetNumberOfLocalDofs( restriction ) == 0 ||
         GetNumberOfGlobalDofs( restriction ) > 0,
      "A nonempty completed restriction must have a nonzero global extent." );
   GENDIL_VERIFY(
      GetNumberOfGlobalDofs( restriction ) <=
         GetAlgebraicDofExtent( restriction ),
      "A completed restriction's logical global DoF count exceeds its algebraic extent." );
   ValidateRestrictionRepresentation( restriction );
}

/**
 * @brief Validate a completed restriction against a particular mesh and
 * finite element.
 */
template < typename Mesh, typename FiniteElement, typename Restriction >
   requires CompatibleElementDoFRestrictionFor<
      Restriction,
      FiniteElement >
void ValidateElementDoFRestrictionFor(
   const Mesh & mesh,
   const FiniteElement &,
   const Restriction & restriction )
{
   ValidateElementDoFRestriction( restriction );

   using ShapeFunctions =
      typename std::remove_cvref_t<
         FiniteElement >::shape_functions;
   const GlobalIndex expected_num_local_dofs =
      CheckedElementLocalDofCount< ShapeFunctions >(
         static_cast< GlobalIndex >( mesh.GetNumberOfCells() ) );
   GENDIL_VERIFY(
      GetNumberOfLocalDofs( restriction ) == expected_num_local_dofs,
      "Completed restriction local DoF count does not match the mesh and finite element." );
}

} // namespace gendil
