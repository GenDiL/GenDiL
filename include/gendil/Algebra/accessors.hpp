// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil
{

// Helper for always_false (reuse existing if available in prelude)
template<class T>
struct always_false : std::false_type {};

// Vector access: v(i) or v[i]
template<class V, class I>
GENDIL_HOST_DEVICE GENDIL_INLINE
decltype(auto) vector_access(V&& v, I i)
{
   if constexpr (requires { std::forward<V>(v)(i); }) {
      return std::forward<V>(v)(i);
   } else if constexpr (requires { std::forward<V>(v)[i]; }) {
      return std::forward<V>(v)[i];
   } else {
      static_assert(always_false<V>::value, "vector_access: unsupported accessor");
   }
}

// Matrix access: A(i,j) or A[i][j]
template<class M, class I, class J>
GENDIL_HOST_DEVICE GENDIL_INLINE
decltype(auto) matrix_access(M&& A, I i, J j)
{
   if constexpr (requires { std::forward<M>(A)(i, j); }) {
      return std::forward<M>(A)(i, j);
   } else if constexpr (requires { std::forward<M>(A)[i][j]; }) {
      return std::forward<M>(A)[i][j];
   } else {
      static_assert(always_false<M>::value, "matrix_access: unsupported accessor");
   }
}

} // namespace gendil
