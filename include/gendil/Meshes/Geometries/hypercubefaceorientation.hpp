// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Shared hypercube face/orientation helpers.
 *
 * These helpers depend on signed-permutation orientation conventions. Pure
 * local-face indexing helpers live on HyperCube.
 */

#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/debug.hpp"

namespace gendil {

/**
 * @brief Recover the native local face reached from a reference face under a
 * signed permutation orientation.
 */
template < Integer Dim >
Integer NativeFaceFromReferenceFace(
   Integer reference_face,
   const Permutation< Dim > & orientation )
{
   const Integer reference_axis =
      HyperCube< Dim >::GetNormalDimensionIndex( reference_face );
   const bool reference_upper = reference_face >= Dim;

   for ( Integer native_axis = 0; native_axis < Dim; ++native_axis )
   {
      const LocalIndex entry = orientation( native_axis );
      const Integer mapped_reference_axis =
         static_cast< Integer >( entry > 0 ? entry - 1 : -entry - 1 );
      if ( mapped_reference_axis != reference_axis )
      {
         continue;
      }

      const bool native_upper = entry > 0 ? reference_upper : !reference_upper;
      return native_upper ? native_axis + Dim : native_axis;
   }

   GENDIL_VERIFY( false, "Invalid hypercube face orientation." );
   return static_cast< Integer >( -1 );
}

} // namespace gendil
