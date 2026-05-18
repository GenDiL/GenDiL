// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Compatibility header.
// Field-expression and form-expression summation machinery has been split
// into separate headers for clarity.
//
// TODO: Consider deprecating this header after all includes migrate to
// sumfieldexpr.hpp and sumformexpr.hpp directly.

#include "gendil/FiniteElementMethod/WeakForm/sumfieldexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"
