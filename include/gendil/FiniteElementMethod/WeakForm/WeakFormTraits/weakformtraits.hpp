// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Describe the fields and mathematical structure of a weak form.
 *
 * These compile-time queries identify what a form reads, which test fields it
 * contributes to, and whether its terms integrate over cells or facets. Operator
 * construction uses this information to validate the form and determine the
 * field values, gradients, and geometry needed for numerical evaluation.
 *
 * Integrand and form signatures bring these properties together, resolving field
 * roles from declarations in the complete form, independently of runtime storage.
 * Simpler expression queries remain useful for inspecting dependencies and
 * constructing pullbacks. The includes build from these queries to signatures;
 * their individual headers document the corresponding contracts.
 */

// Describe whether a field's values, gradients, or both are required.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/operatormask.hpp"

// Associate field names with their value/gradient requirements and origins.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/namedrequirements.hpp"

// Identify symbolic integration domains and cell, interior, or boundary contributions.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/domaintraits.hpp"

// Classify test dependence for pullback construction and direct trace evaluation.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"

// Inspect field-use origins, explicit trace dependencies, and plus-side geometry needs.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/fielddependencies.hpp"

// Describe integration sides, expression shapes, affinity, and geometry requirements.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/semantictypes.hpp"

// Collect field declarations and retain relationships between inputs and domains.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/formsymbols.hpp"

// Analyze expression dependencies, test linearity, and active-field affinity.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/expressionsemantics.hpp"

// Summarize integrands and forms, including reads, outputs, zeros, and domain/entity selection.
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/signatures.hpp"
