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

// Element-side normal and surface density at a facet quadrature point.
// The reference cell is [0,1]^d, so ||J^{-T} n_ref|| is the inverse
// normal cell size. It can differ across an interior facet and is not the
// inverse characteristic length of the shared physical subfacet.
// Multiply det_J_facet by the reference subfacet measure for integration.

template <Integer Dim>
struct FacetGeometry
{
   std::array<Real, Dim> normalized_physical_normal;
   Real inverse_normal_cell_size;
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

   Real inverse_normal_cell_size_sq = 0.0;
   for (Integer i = 0; i < Dim; ++i)
   {
      inverse_normal_cell_size_sq += n_unnormalized_array[i] * n_unnormalized_array[i];
   }
   Real inverse_normal_cell_size = std::sqrt(inverse_normal_cell_size_sq);

   std::array<Real, Dim> normalized_physical_normal{};
   for (Integer i = 0; i < Dim; ++i)
   {
      normalized_physical_normal[i] = n_unnormalized_array[i] / inverse_normal_cell_size;
   }

   Real det_J_facet = det_J * inverse_normal_cell_size;

   return FacetGeometry<Dim>{
      normalized_physical_normal,
      inverse_normal_cell_size,
      det_J_facet
   };
}

} // namespace gendil
