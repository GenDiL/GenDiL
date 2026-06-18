// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


// Structs to represent a finite element.
#include "gendil/FiniteElementMethod/finiteelement.hpp"

// Collection of shape functions.
#include "gendil/FiniteElementMethod/ShapeFunctions/shapefunctions.hpp"

// A generic high-dimension finite element space inheriting from a Mesh and a FiniteElement.
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"

// Homogeneous finite element spaces over Cartesian global face meshes.
#include "gendil/FiniteElementMethod/globalfacefiniteelementspace.hpp"

// Lightweight aggregation of homogeneous finite element spaces.
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"

#include "gendil/Utilities/KernelContext/batchingeligibility.hpp"

// An expression template domain specific language (DSL) to represent weak forms.
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"

#include "gendil/FiniteElementMethod/MatrixAssembly/assembly.hpp"
