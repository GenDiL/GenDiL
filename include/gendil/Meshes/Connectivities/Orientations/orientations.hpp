// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Complete public interface for mesh and tensor-axis orientations.
 *
 * This umbrella exposes flat runtime permutations, compile-time identity
 * orientations, compact tensor-product orientations, and both directions of
 * reference/native tensor-index conversion. Consumers that need only one
 * facility may include the corresponding focused header instead.
 */

// Flat runtime signed permutations and their general traits and utilities.
#include "permutation.hpp"

// Compile-time identity orientations.
#include "identityorientation.hpp"

// Compact factor-preserving tensor-product orientations.
#include "tensorproductorientation.hpp"

// Reference-to-native tensor-index conversion.
#include "referencetonativeindex.hpp"

// Native-to-reference tensor-index conversion.
#include "nativetoreferenceindex.hpp"
