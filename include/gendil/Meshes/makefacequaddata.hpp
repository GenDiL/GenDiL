// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"

namespace gendil {

/**
 * @brief Construct the mesh face quadrature data.
 * 
 * @tparam Mesh The mesh type.
 * @tparam FaceIntegrationRules A tuple of integration rule types for each face.
 * @return auto A tuple of mesh quadrature data for each face.
 */
template < typename Mesh, typename ... FaceIntegrationRules >
auto MakeMeshFaceQuadData()
{
   return std::make_tuple( typename Mesh::cell_type::template QuadData< FaceIntegrationRules >{}... );
}

template < typename Mesh, typename ... FaceIntegrationRules >
auto MakeMeshFaceQuadData( std::tuple< FaceIntegrationRules... > )
{
   return MakeMeshFaceQuadData< Mesh, FaceIntegrationRules... >();
}

}
