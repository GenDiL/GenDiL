// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil
{

/// @brief device friendly swap (not needed with C++20(?))
template < typename T >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr void Swap(T& x, T& y)
{
   T z(std::move(x));
   x = std::move(y);
   y = std::move(z);
}

} // namespace gendil
