// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


// Finite element functions used to build a finite element operator.
#include "KernelOperators/kerneloperators.hpp"

// Class helping computation of linear forms.
#include "linearformoperator.hpp"

// Base class for matrix-free bilinear finite element operators.
#include "bilinearoperator.hpp"

// Base class for matrix-free mixed bilinear finite element operators.
#include "mixedbilinearoperator.hpp"

// A high-dimension mass operator.
#include "massoperator.hpp"

// A high-dimension L2 mass inverse operator.
#include "massinverseoperator.hpp"

// A high-dimension mass operator with different finite element space for trial and test spaces.
#include "mixedmassoperator.hpp"

// A high-dimension conservative DG advection operator.
#include "advectionoperator.hpp"

// A high-dimension conservative DG mass+advection operator.
#include "massadvectionoperator.hpp"

// A high-dimension SIPDG diffusion operator.
#include "diffusionoperator.hpp"

// An operator to measure experimental "speed of light" for a matrix-free operator without face terms.
#include "speedoflightoperator.hpp"

// An operator to measure experimental "speed of light" for a matrix-free operator with face terms.
#include "facespeedoflightoperator.hpp"

// An operator to compute the L2 error.
#include "L2error.hpp"

// An operator to project a high dimension field onto the first subspace using L2 norm.
#include "L2projectionoperator.hpp"

// An operator to project a high dimension field onto the first subspace using Linf norm.
#include "Linfprojectionoperator.hpp"

// A grad-grad operator.
#include "gradgradoperator.hpp"

// A continuum-kinetic weak coupling using upwind fluxes.
#include "eulervlasovoperator.hpp"
