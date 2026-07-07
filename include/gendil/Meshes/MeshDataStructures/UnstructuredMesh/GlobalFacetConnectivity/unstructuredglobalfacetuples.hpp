// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Public tuple aliases for heterogeneous unstructured global-face
 * connectivity.
 *
 * A concrete connectivity type represents one statically typed execution
 * family. These tuple aliases collect every static face family for conforming
 * interior, nonconforming interior, or boundary domains.
 */

#include <tuple>
#include <utility>

#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructuredboundaryfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructuredinteriorfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructurednonconforminginteriorfaceconnectivity.hpp"

namespace gendil {

template < typename Geometry, Integer... I >
auto UnstructuredConformingInteriorConnectivityTupleType(
   std::integer_sequence< Integer, I... > )
   -> std::tuple< UnstructuredInteriorFaceConnectivity< Geometry, I >... >;

template < typename Geometry >
/**
 * @brief Tuple containing every static conforming interior face family for an
 * unstructured Geometry.
 */
using UnstructuredConformingInteriorConnectivityTuple =
   decltype( UnstructuredConformingInteriorConnectivityTupleType< Geometry >(
      std::make_integer_sequence< Integer, Geometry::num_faces >{} ) );

template < typename Geometry, Integer... I >
auto UnstructuredNonconformingInteriorConnectivityTupleType(
   std::integer_sequence< Integer, I... > )
   -> std::tuple< UnstructuredNonconformingInteriorFaceConnectivity< Geometry, I >... >;

template < typename Geometry >
/**
 * @brief Tuple containing every static nonconforming interior leaf-face family
 * for an unstructured Geometry.
 */
using UnstructuredNonconformingInteriorConnectivityTuple =
   decltype( UnstructuredNonconformingInteriorConnectivityTupleType< Geometry >(
      std::make_integer_sequence< Integer, Geometry::num_faces >{} ) );

template < typename Geometry, Integer... I >
auto UnstructuredBoundaryFaceConnectivityTupleType(
   std::integer_sequence< Integer, I... > )
   -> std::tuple< UnstructuredBoundaryFaceConnectivity< Geometry, I >... >;

template < typename Geometry >
/**
 * @brief Tuple containing every static boundary face family for an
 * unstructured Geometry.
 */
using UnstructuredBoundaryFaceConnectivityTuple =
   decltype( UnstructuredBoundaryFaceConnectivityTupleType< Geometry >(
      std::make_integer_sequence< Integer, Geometry::num_faces >{} ) );

} // namespace gendil
