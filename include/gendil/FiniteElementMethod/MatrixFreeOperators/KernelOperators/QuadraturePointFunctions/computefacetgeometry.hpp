// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/computephysicalnormal.hpp"
#include <cmath>
#include <array>

namespace gendil {

// =============================================================================
// ComputeFacetGeometry: Cleaned-up facet geometry
// =============================================================================
//
// **Purpose:**
//    Compute cleaned-up facet geometry quantities at a quadrature point:
//      - normalized physical normal (unit length)
//      - InverseFacetSize (local facet diameter inverse)
//      - det_J_facet (surface Jacobian determinant)
//
// **Design:**
//    This implements the cleaned-up facet convention from audit 2B.7A:
//
//      n_unnormalized = J^{-T} * n_ref
//      InverseFacetSize = norm(n_unnormalized)
//      n_physical = n_unnormalized / InverseFacetSize
//      det_J_facet = det_J * InverseFacetSize
//
//    where det_J is the current volume Jacobian determinant.
//
// **Reference Element:**
//    GenDiL uses [0,1]^d reference element (not [-1,1]^d).
//    For Cartesian mesh with spacing h: J = h * I, det_J = h^d, inv_J = (1/h) * I
//
// **Rationale:**
//    The old facet path uses volume det_J with unnormalized normal.
//    The normal's magnitude implicitly carries h^{-1}, compensating for
//    using volume measure instead of surface measure on Cartesian meshes.
//
//    This is a hack that:
//      - requires knowing h a priori for penalty coefficients;
//      - conflates measure and normal direction;
//      - only works cleanly on Cartesian/affine meshes.
//
//    The cleaned-up convention separates these roles:
//      - normalized normal provides direction only (unit length);
//      - InverseFacetSize provides local facet scale explicitly (= 1/h for Cartesian);
//      - det_J_facet provides proper surface measure.
//
//    SIPDG penalty scaling becomes:
//      tau ~ kappa * InverseFacetSize  (standard h^{-1})
//    instead of the old:
//      tau ~ kappa * h^{-2}  (extra compensation)
//
// **Quadrature-Point Concept:**
//    InverseFacetSize is computed per quadrature point.
//    For affine/tensor-product elements, it is constant across the face.
//    For curved/high-order elements, it may vary by quadrature point.
//
// **Future CoefficientInput:**
//    InverseFacetSize should eventually become a first-class CoefficientInput,
//    enabling weak forms to express geometry-dependent coefficients:
//
//      auto tau = penalty_factor * kappa * InverseFacetSize;
//
//    This is required for general unstructured/nonuniform/curved meshes
//    where penalty coefficients cannot be pre-computed constants.
//
// **Returns:**
//    Struct with:
//      - normalized_physical_normal: std::array<Real, Dim> (unit length)
//      - inverse_facet_size: Real (norm of unnormalized normal)
//      - det_J_facet: Real (surface Jacobian determinant)
//
// **Input:**
//    - inv_J: Inverse Jacobian (element's volume Jacobian inverse)
//    - reference_normal: CanonicalVector<Dim, Index, Sign> from face
//    - det_J: Volume Jacobian determinant
//
// =============================================================================

template <Integer Dim>
struct FacetGeometry
{
   std::array<Real, Dim> normalized_physical_normal;
   Real inverse_facet_size;
   Real det_J_facet;
};

template <typename InvJacobian, Integer Dim, Integer Index, int Sign>
GENDIL_HOST_DEVICE
FacetGeometry<Dim> ComputeFacetGeometry(
   InvJacobian const & inv_J,
   CanonicalVector<Dim, Index, Sign> reference_normal,
   Real det_J)
{
   // Compute unnormalized physical normal using existing function
   // This handles both single matrix and std::tuple<MatrixTypes...>
   auto n_unnormalized_array = ComputePhysicalNormal(inv_J, reference_normal);

   // Compute InverseFacetSize = norm(n_unnormalized)
   // For affine/tensor-product elements, this is the inverse of the
   // local facet diameter/mesh size h_F
   Real inverse_facet_size_sq = 0.0;
   for (Integer i = 0; i < Dim; ++i)
   {
      inverse_facet_size_sq += n_unnormalized_array[i] * n_unnormalized_array[i];
   }
   Real inverse_facet_size = std::sqrt(inverse_facet_size_sq);

   // Compute normalized physical normal: n_physical = n_unnormalized / InverseFacetSize
   // This gives a unit-length normal vector
   std::array<Real, Dim> normalized_physical_normal{};
   for (Integer i = 0; i < Dim; ++i)
   {
      normalized_physical_normal[i] = n_unnormalized_array[i] / inverse_facet_size;
   }

   // Compute surface Jacobian determinant: det_J_facet = det_J * InverseFacetSize
   // This converts the volume determinant to a surface determinant
   Real det_J_facet = det_J * inverse_facet_size;

   return FacetGeometry<Dim>{
      normalized_physical_normal,
      inverse_facet_size,
      det_J_facet
   };
}

} // namespace gendil
