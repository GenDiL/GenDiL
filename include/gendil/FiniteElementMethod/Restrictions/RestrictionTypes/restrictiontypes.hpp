// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Unit coefficient emitted by built-in unit-weight restriction rows.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionweight.hpp"

// Scalar completed restrictions with compile-time row weights and runtime indices.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/fixedstencilrestriction.hpp"

// Compile-time aggregation of component restriction maps.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/vectorrestriction.hpp"

// Runtime row-orientation signs applied to completed restrictions.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/signedrestriction.hpp"

// Completed contiguous L2 restrictions and their construction specifications.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/contiguousl2restriction.hpp"

// Completed indirect H1 restrictions and their construction specifications.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/indirecth1restriction.hpp"

// Map-free nodal H1 restrictions for Cartesian meshes.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/cartesianh1restriction.hpp"

// Completed tensor-product restrictions and factor addressing.
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/tensorproductrestriction.hpp"
