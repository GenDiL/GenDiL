// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil
{

template<class FiniteElementSpace, class InteriorFaceMesh>
struct GlobalInteriorFaceFiniteElementSpace
{
   FiniteElementSpace finite_element_space;
   InteriorFaceMesh face_mesh;
};

template<class FiniteElementSpace, class InteriorFaceMesh>
constexpr auto MakeGlobalInteriorFaceFiniteElementSpace(
   const FiniteElementSpace& finite_element_space,
   const InteriorFaceMesh& face_mesh)
{
   return GlobalInteriorFaceFiniteElementSpace<
      FiniteElementSpace,
      InteriorFaceMesh>{ finite_element_space, face_mesh };
}

template<class FiniteElementSpace, class BoundaryFaceMesh>
struct GlobalBoundaryFaceFiniteElementSpace
{
   FiniteElementSpace finite_element_space;
   BoundaryFaceMesh face_mesh;
};

template<class FiniteElementSpace, class BoundaryFaceMesh>
constexpr auto MakeGlobalBoundaryFaceFiniteElementSpace(
   const FiniteElementSpace& finite_element_space,
   const BoundaryFaceMesh& face_mesh)
{
   return GlobalBoundaryFaceFiniteElementSpace<
      FiniteElementSpace,
      BoundaryFaceMesh>{ finite_element_space, face_mesh };
}

} // namespace gendil
