// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


// A container to store the degrees of freedom of one element.
#include "elementdof.hpp"

// A container to store all the quadrature values of a field based on an integration rule.
#include "quadraturepointvalues.hpp"

// Simple structs representing shape functions evaluated at quadrature points.
#include "doftoquad.hpp"

// Kernel operators to apply on the trial space to obtain field at quadrature points.
#include "TrialSpaceOperators/trialspaceoperators.hpp"

// Kernel operators to apply on the test space to obtain field contributions from quadrature points values.
#include "TestSpaceOperators/testspaceoperators.hpp"

// Helpers to help iterate over indices for different quantities (DoF, quads, faces, etc...)
#include "LoopHelpers/loophelpers.hpp"

// Functions to be used inside a quadrature loop
#include "QuadraturePointFunctions/quadraturepointfunctions.hpp"

// Helper functions to manage global DoF and local DoF.
#include "DoFIO/dofio.hpp"

// Helper functions to manage quadrature point values
#include "QuadraturePointIO/quadraturepointio.hpp"

