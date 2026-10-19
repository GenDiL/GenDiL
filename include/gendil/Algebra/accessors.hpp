// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include <utility>

namespace gendil
{

// Helper for always_false (reuse existing if available in prelude)
template<class T>
struct always_false : std::false_type {};

// Read or write a vector entry with GetValue(v, i), preferring v(i) to v[i].
// Preserve references/proxies and reject unsupported access in requires expressions.
template<class V, class I>
   requires (requires(V&& v, I i) { std::forward<V>(v)(i); } ||
             requires(V&& v, I i) { std::forward<V>(v)[i]; })
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr decltype(auto) GetValue(V&& v, I i)
{
   if constexpr (requires { std::forward<V>(v)(i); }) {
      return std::forward<V>(v)(i);
   } else {
      return std::forward<V>(v)[i];
   }
}

// Read or write a matrix entry with GetValue(A, i, j), preferring A(i,j) to A[i][j].
// Preserve references/proxies, including writable entries for mutable inputs.
template<class M, class I, class J>
   requires (requires(M&& A, I i, J j) { std::forward<M>(A)(i, j); } ||
             requires(M&& A, I i, J j) { std::forward<M>(A)[i][j]; })
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr decltype(auto) GetValue(M&& A, I i, J j)
{
   if constexpr (requires { std::forward<M>(A)(i, j); }) {
      return std::forward<M>(A)(i, j);
   } else {
      return std::forward<M>(A)[i][j];
   }
}

} // namespace gendil
