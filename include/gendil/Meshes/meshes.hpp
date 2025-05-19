// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


// The common functions to all meshes
#include "mesh.hpp"

// The common functions to all cells
#include "cell.hpp"

// An unstructured 1d mesh using mfem::Mesh nodes
#include "MeshDataStructures/UnstructuredMesh/linemesh.hpp"

// A Cartesian 1d mesh
#include "MeshDataStructures/CartesianMesh/cartesian1dmesh.hpp"

// A unstructured 2d quad mesh using mfem::Mesh nodes
#include "MeshDataStructures/UnstructuredMesh/quadmesh.hpp"

// A 2d Cartesian mesh
#include "MeshDataStructures/CartesianMesh/cartesian2dmesh.hpp"

// An unstructured 3d hex mesh using mfem::Mesh nodes
#include "MeshDataStructures/UnstructuredMesh/hexmesh.hpp"

// A 3d Cartesian mesh
#include "MeshDataStructures/CartesianMesh/cartesian3dmesh.hpp"

// A Cartesian product mesh
#include "MeshDataStructures/TensorProductMesh/productmesh.hpp"

// An arbitrary dimension Cartesian mesh
#include "MeshDataStructures/CartesianMesh/cartesianmesh.hpp"
#include "MeshDataStructures/CartesianMesh/parallelcartesianmesh.hpp"
