// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil
{

// Value-level inner product (Frobenius) for rank-2 SerialRecursiveArray
template<typename T, Integer M, Integer N>
GENDIL_HOST_DEVICE
T Inner(const SerialRecursiveArray<T, M, N>& A, const SerialRecursiveArray<T, M, N>& B)
{
   T result = T{0};
   for (Integer i = 0; i < M; ++i) {
      for (Integer j = 0; j < N; ++j) {
         result += A(i, j) * B(i, j);
      }
   }
   return result;
}

} // namespace gendil
