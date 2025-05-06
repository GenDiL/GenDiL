// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

// Arbitrary dimensional loop abstraction.
#include "loop.hpp"

// Nested loop using thread indices for loop indices.
#include "threadloop.hpp"

// Like loop with the exception that a zero dimensional loop will also run.
#include "unitloop.hpp"

// Arbitrary dimensional loop abstraction which is guaranteed to unroll and
// whose loop indices are compile time constants.
#include "constexprloop.hpp"
