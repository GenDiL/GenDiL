// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


// Finite-element representations and scalar, vector, and tensor composition.
#include "gendil/FiniteElementMethod/FiniteElements/finiteelements.hpp"

// Collection of shape functions.
#include "gendil/FiniteElementMethod/ShapeFunctions/shapefunctions.hpp"

// Element-local to algebraic DoF restriction abstractions.
#include "gendil/FiniteElementMethod/Restrictions/restrictions.hpp"

// A generic high-dimension finite element space inheriting from a Mesh and a FiniteElement.
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"

// Lightweight aggregation of homogeneous finite element spaces.
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"

// Borrowed scalar component views of homogeneous and mixed vector spaces.
#include "gendil/FiniteElementMethod/componentfiniteelementspace.hpp"

// Tensor-product meshes, restrictions, and spaces built from factor spaces.
#include "gendil/FiniteElementMethod/tensorproductfiniteelementspace.hpp"

// An expression template domain specific language (DSL) to represent weak forms.
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"

#include "gendil/FiniteElementMethod/MatrixAssembly/assembly.hpp"
