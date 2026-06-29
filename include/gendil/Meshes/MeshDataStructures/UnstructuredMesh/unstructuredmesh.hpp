// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// A 1d unstructured mesh.
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/linemesh.hpp"

// A 2d unstructured quad mesh.
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/quadmesh.hpp"

// A 3d unstructured hex mesh.
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/hexmesh.hpp"

// Cell-only unstructured meshes for global-facet workflows.
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/cellmesh.hpp"

// LocalFacetConnectivity: local/cell-neighbor facet connectivity for
// unstructured meshes.
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/LocalFacetConnectivity/unstructuredconformingconnectivity.hpp"

// GlobalFacetConnectivity: materialized global facet execution families and
// restricted fallback construction.
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/GlobalFacetConnectivity/unstructuredglobalfaceconnectivity.hpp"
