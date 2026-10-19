// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Express finite-element weak forms using named fields and integrals.
 *
 * Combine trial and test fields, coefficients, and differential or facet
 * operators to describe an integrand. integrate associates that expression
 * with a named cell or facet domain; adding integrals builds a complete form.
 * lhs and rhs extract the linear operator and load from an affine residual.
 *
 * The form describes the mathematics. A WeakFormContext connects its names to
 * finite-element spaces, persistent field data, and integration domains for
 * operator construction. The included traits describe field dependencies and
 * validate the form's mathematical structure. Individual headers document the
 * construction syntax and supported operations.
 */

// Define the expression categories and scalar, vector, and matrix field shapes.
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

// Define named cell and facet domain types for integration and analysis.
#include "gendil/FiniteElementMethod/WeakForm/domain.hpp"

// Declare named trial fields whose values come from the operator's active input.
#include "gendil/FiniteElementMethod/WeakForm/trialspace.hpp"

// Declare named test fields that identify the residual's output spaces.
#include "gendil/FiniteElementMethod/WeakForm/testspace.hpp"

// Refer to named finite-element field values in expressions.
#include "gendil/FiniteElementMethod/WeakForm/finiteelementfield.hpp"

// Describe the field and geometry inputs requested by coefficient callbacks.
#include "gendil/FiniteElementMethod/WeakForm/coefficientinputs.hpp"

// Define coefficient callbacks and extract their requested inputs at evaluation.
#include "gendil/FiniteElementMethod/WeakForm/coefficient.hpp"

// Use the physical normal in facet expressions.
#include "gendil/FiniteElementMethod/WeakForm/normal.hpp"

// Negate a field expression.
#include "gendil/FiniteElementMethod/WeakForm/neg.hpp"

// Add and subtract field expressions within an integrand.
#include "gendil/FiniteElementMethod/WeakForm/sumfieldexpr.hpp"

// Combine integrals while preserving their order and separate evaluation.
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"

// Provide multiplication syntax for operands without a classified algebraic product.
#include "gendil/FiniteElementMethod/WeakForm/mult.hpp"

// Scale a field expression by a real constant.
#include "gendil/FiniteElementMethod/WeakForm/scale.hpp"

// Form the dot product of vector expressions.
#include "gendil/FiniteElementMethod/WeakForm/dot.hpp"

// Form inner products of matching vector or matrix expressions.
#include "gendil/FiniteElementMethod/WeakForm/inner.hpp"

// Form the outer product of vector expressions.
#include "gendil/FiniteElementMethod/WeakForm/outer.hpp"

// Select the minus or plus trace of an interior-facet expression.
#include "gendil/FiniteElementMethod/WeakForm/trace.hpp"

// Form jumps across interior facets, with one-sided physical-boundary semantics.
#include "gendil/FiniteElementMethod/WeakForm/jump.hpp"

// Form averages across interior facets, with one-sided physical-boundary semantics.
#include "gendil/FiniteElementMethod/WeakForm/average.hpp"

// Form upwind normal fluxes on interiors or boundaries with explicit inflow data.
#include "gendil/FiniteElementMethod/WeakForm/upwind.hpp"

// Request spatial gradients of supported field expressions.
#include "gendil/FiniteElementMethod/WeakForm/gradient.hpp"

// Form physical divergence of named vector trial, test, or prescribed fields.
#include "gendil/FiniteElementMethod/WeakForm/divergence.hpp"

// Form three-dimensional curl of named vector trial, test, or prescribed fields.
#include "gendil/FiniteElementMethod/WeakForm/curl.hpp"

// Transpose field expressions, including vector-to-row-matrix conversion.
#include "gendil/FiniteElementMethod/WeakForm/transpose.hpp"

// Classify multiplication from the operands' scalar, vector, or matrix shapes.
#include "gendil/FiniteElementMethod/WeakForm/productkind.hpp"

// Construct scalar multiplication, matrix-vector products, and matrix-matrix products.
#include "gendil/FiniteElementMethod/WeakForm/productexpr.hpp"

// Associate integrands with named cell, interior-facet, or boundary-facet domains.
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"

// Register named domains, finite-element spaces, and persistent field data.
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"

// Analyze the form's field requirements, integration entities, and signatures.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/weakformtraits.hpp"

// Classify dependence on test fields for validation and pullback construction.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"

// Determine the scalar, vector, or matrix shape of field expressions.
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"

// Extract lhs and rhs forms from residuals with provably affine trial dependence.
#include "gendil/FiniteElementMethod/WeakForm/affinesplit.hpp"
