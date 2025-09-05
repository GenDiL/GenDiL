// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// A 1d Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesian1dmesh.hpp"

// A 2d Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesian2dmesh.hpp"

// A 3d Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesian3dmesh.hpp"

// An arbitrary dimension Cartesian mesh

// A face connectivity describing local face connectivity on a Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianlocalfaceconnectivity.hpp"

// A face connectivity describing local face connectivity on a periodic Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/periodiccartesianconnectivity.hpp"

// A face connectivity describing global face connectivity on a Cartesian mesh
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianglobalfaceconnectivity.hpp"

// An intermesh face connectivity between two conforming Cartesian meshes
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianintermeshfaceconnectivity.hpp"

// An intermesh face connectivity between two nonconforming Cartesian meshes
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/nonconformingcartesianintermeshfaceconnectivity.hpp"
