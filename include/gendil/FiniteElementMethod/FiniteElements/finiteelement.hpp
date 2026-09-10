// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Core finite-element representation.
 */

#include "gendil/Utilities/types.hpp"

namespace gendil {

/**
 * @brief Pair a reference geometry with its local shape-function collection.
 *
 * A finite element describes only element-local approximation data. The mesh
 * and local-to-global algebraic mapping belong to a finite-element space.
 *
 * @tparam Geometry Reference geometry and topology of the element.
 * @tparam ShapeFunctions Shape functions defined on that geometry.
 */
template<typename Geometry, typename ShapeFunctions>
struct FiniteElement
{
   using geometry = Geometry;
   using shape_functions = ShapeFunctions;
   static constexpr Integer space_dim = Geometry::space_dim;
   static constexpr Integer geometry_dim = Geometry::geometry_dim;

   constexpr FiniteElement()
   {
      static_assert(
         geometry_dim == shape_functions::dim,
         "Element topology dimension and element shape functions dimension mismatch."
      );
   }

   /** Return the total number of element-local DoFs. */
   static constexpr Integer GetNumDofs()
   {
      return shape_functions::num_dofs;
   }
};

} // namespace gendil
