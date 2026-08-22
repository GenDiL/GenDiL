// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Structural, finite-element compatibility, and contextual restriction concepts.
#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"

// Restriction capability and representation-family traits.
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"

// Generic gather and adjoint-scatter operations.
#include "gendil/FiniteElementMethod/Restrictions/restrictionoperations.hpp"

// Generic completed-restriction invariant validation.
#include "gendil/FiniteElementMethod/Restrictions/restrictionvalidation.hpp"

// Finite-element DoF shapes, local counts, offsets, and flattening utilities.
#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"

// Singular global DoF lookup for restrictions and finite-element spaces.
#include "gendil/FiniteElementMethod/Restrictions/globaldofindex.hpp"

// Default tensor-leaf view layout backed by semantic restriction lookup.
#include "gendil/FiniteElementMethod/Restrictions/restrictionlayout.hpp"

// Built-in scalar, vector, and tensor-product restriction representations.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictiontypes.hpp"
