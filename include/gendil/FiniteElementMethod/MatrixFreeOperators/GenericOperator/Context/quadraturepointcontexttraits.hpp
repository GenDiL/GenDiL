// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil
{

// Identify physical-boundary evaluation without depending on concrete contexts.
// A selected interior side is also one-sided, but is not a physical boundary.
// The explicit marker guards trace evaluation and pullback geometry compatibility.
template < typename Context >
inline constexpr bool is_physical_boundary_v = []
{
   if constexpr ( requires { Context::is_physical_boundary; } )
   {
      return bool(Context::is_physical_boundary);
   }
   else
   {
      return false;
   }
}();

} // namespace gendil
