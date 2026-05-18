// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil
{

/**
 * @brief Value-level outer product for rank-1 SerialRecursiveArray vectors.
 *
 * Computes the outer product of two vectors:
 *    result[c][j] = a[c] * b[j]
 *
 * **Orientation:** Component-first indexing, matching grad(U) tensor layout.
 *    result(component, spatial_dim) = a(component) * b(spatial_dim)
 *
 * This matches the gradient orientation used in GenDiL:
 *    grad(u)(component, direction) = ∂u_component/∂x_direction
 *
 * @tparam T Element type (typically Real)
 * @tparam M First dimension (number of components)
 * @tparam N Second dimension (spatial dimension)
 * @param a First vector (field components)
 * @param b Second vector (spatial vector, e.g., normal)
 * @return SerialRecursiveArray<T, M, N> outer product matrix
 */
template<typename T, Integer M, Integer N>
GENDIL_HOST_DEVICE
SerialRecursiveArray<T, M, N> Outer(
   const SerialRecursiveArray<T, M>& a,
   const SerialRecursiveArray<T, N>& b)
{
   SerialRecursiveArray<T, M, N> result;

   for (Integer i = 0; i < M; ++i) {
      for (Integer j = 0; j < N; ++j) {
         result(i, j) = a(i) * b(j);
      }
   }

   return result;
}

/**
 * @brief Value-level outer product for RecursiveArray and std::array.
 *
 * Computes result[c][j] = a[c] * b[j]
 *
 * @tparam T Element type (typically Real)
 * @tparam M First dimension (number of components)
 * @tparam N Second dimension (spatial dimension)
 * @param a First vector (field components, RecursiveArray)
 * @param b Second vector (spatial vector, std::array - e.g., normal)
 * @return SerialRecursiveArray<T, M, N> outer product matrix
 */
template<typename T, Integer M, size_t N>
GENDIL_HOST_DEVICE
SerialRecursiveArray<T, M, static_cast<Integer>(N)> Outer(
   const SerialRecursiveArray<T, M>& a,
   const std::array<T, N>& b)
{
   SerialRecursiveArray<T, M, static_cast<Integer>(N)> result;

   for (Integer i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
         result(i, static_cast<Integer>(j)) = a(i) * b[j];
      }
   }

   return result;
}

} // namespace gendil
