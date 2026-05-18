// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Utilities/toarray.hpp"

namespace gendil {

template < typename... MatrixTypes, Integer Dim, Integer Index, int Sign >
GENDIL_HOST_DEVICE
auto ComputePhysicalNormal(
   std::tuple< MatrixTypes... > const & inv_J,
   CanonicalVector< Dim, Index, Sign > )
{
   Real res[ Dim ]{};
   res[ Index ] = Sign;
   ApplyMapping( inv_J, res );
   return to_array( res );
}

template < typename MatrixType, Integer Dim, Integer Index, int Sign >
GENDIL_HOST_DEVICE
auto ComputePhysicalNormal(
   MatrixType const & inv_J,
   CanonicalVector< Dim, Index, Sign > )
{
   Real res[ Dim ]{};
   res[ Index ] = Sign;
   ApplyMapping( inv_J, res );
   return to_array( res );
}

}
