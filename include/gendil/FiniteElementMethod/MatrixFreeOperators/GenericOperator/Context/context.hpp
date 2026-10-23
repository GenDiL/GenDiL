// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Carry the information needed at each stage of a matrix-free application.
 *
 * Applying a matrix-free operator combines a weak form, finite-element fields,
 * vector data, and mesh geometry. Context objects bring the relevant information
 * together at different levels, from the operator's named fields and domains
 * down to the geometry at one quadrature point.
 *
 * WeakFormContext registers domains, spaces, and persistent field data. Selection
 * narrows those registrations to one batch of cells or faces. Operator contexts
 * provide reusable basis and geometry preparation for that batch, while
 * ExecutionContext connects its fields to the input and output of one invocation.
 * These complement each other: numerical routines
 * receive both preparation and storage bindings, keeping preparation independent
 * of the vectors being used.
 *
 * During traversal, ElementContext identifies the current cell. Facet contexts
 * add the neighboring geometry needed for interior-face evaluation. Quadrature
 * point contexts then supply coordinates, geometric transformations, weights,
 * and normals as appropriate for evaluating the integrand at one point.
 *
 * The includes follow these levels, with construction and validation helpers last.
 */

// Register the operator's named domains, finite-element spaces, and persistent data.
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"

// Select the required spaces and persistent fields for one cell or face batch.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/restrictedweakformcontext.hpp"

// Provide reusable preparation and use its signatures to bind local element rows.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"

// Specialize preparation for cell traversal and the cells' local facets.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/localoperatorcontext.hpp"

// Specialize preparation for batches of global interior or boundary faces.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globaloperatorcontext.hpp"

// Connect selected fields to this invocation's vectors or borrowed element rows.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/executioncontext.hpp"

// Identify the current element and its cell geometry during traversal.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/elementcontext.hpp"

// Add oriented neighbor geometry when evaluating an interior face.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/facetcontext.hpp"

// Supply geometry, integration weights, and normals at the current quadrature point.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontext.hpp"

// Identify physical-boundary evaluation independently of concrete context types.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontexttraits.hpp"

// Supporting selection, validation, and preparation.

// Describe a selected batch of cells or faces and its adjacent meshes.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/selectedexecutiondomain.hpp"

// Associate each face batch with its adjacent finite-element spaces.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/globalfacefieldbinding.hpp"

// Select compatible field restrictions for that batch.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/fieldselection.hpp"

// Describe named field access for selection and invocation binding.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/fieldrequirements.hpp"

// Translate resolved weak-form signatures into named read/write requirements.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformrequirements.hpp"

// Validate the form's domains, field shapes, retained dimensions, and bound storage.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/weakformvalidation.hpp"

// Share cell preparation, signature lookup, and geometry checks between operator contexts.
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontextcommon.hpp"
