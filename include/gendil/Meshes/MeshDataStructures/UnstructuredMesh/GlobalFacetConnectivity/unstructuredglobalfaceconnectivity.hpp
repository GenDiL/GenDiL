// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Public unstructured global-face connectivity components.
 *
 * A concrete global-face connectivity type represents one statically typed
 * execution family, not an entire heterogeneous topology. Construction
 * materializes a snapshot in GenDiL-managed host/device storage. Connectivity
 * copies share that storage. No source-mesh pointer is retained.
 *
 * The restricted local-connectivity fallback lives in
 * globalfacesfromlocalconnectivity.hpp and is intentionally not included by
 * this public umbrella. It belongs to this folder because it emits global
 * facet execution families.
 */

#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructuredinteriorfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructurednonconforminginteriorfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructuredboundaryfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructuredglobalfacetuples.hpp"
