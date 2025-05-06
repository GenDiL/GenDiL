// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

/**
 * @brief A structure representing a canonical vector in arbitrary dimension.
 * 
 * @tparam Dim The dimension of the space/
 * @tparam Index The index of the canonical vector.
 * @tparam Sign The sign of the canonical vector.
 */
template < Integer Dim, Integer Index, int Sign >
struct CanonicalVector
{
   static constexpr Integer dim = Dim;
   static constexpr Integer index = Index;
   static constexpr int sign = Sign; // -1 or 1
};

}
