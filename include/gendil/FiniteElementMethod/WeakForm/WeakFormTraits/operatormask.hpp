// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once


namespace gendil
{

enum class OperatorMask : unsigned
{
   None = 0u,
   Values = 1u,
   Gradients = 2u
};

constexpr OperatorMask operator+(OperatorMask a, OperatorMask b) {
   return OperatorMask(unsigned(a) | unsigned(b));
}

constexpr bool need_values(OperatorMask m) { return (unsigned(m) & 1u) != 0u; }
constexpr bool need_gradients(OperatorMask m) { return (unsigned(m) & 2u) != 0u; }

} // namespace gendil
