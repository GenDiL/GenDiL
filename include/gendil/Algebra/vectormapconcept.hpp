// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Callable protocol and concept for sized maps between vectors.
 *
 * A vector map represents a possibly nonlinear mapping from an input vector
 * space to an output vector space. A vector map is callable for replacement
 * application and also provides the free functions `GetInputSize`,
 * `GetOutputSize`, `Apply`, and `ApplyAdd`. The overloads below adapt the
 * conventional GenDiL member interface; callable types may instead customize
 * the free functions in their associated namespace for ADL.
 *
 * `VectorMap` is contextual in its input and output vector types. It does not
 * imply linearity, invertibility, ownership, alias safety, or matching runtime
 * dimensions. The forwarding adapters are host/device callable when the
 * underlying map operations are; the concept does not check device callability.
 */

#include "gendil/Utilities/types.hpp"

#include <concepts>
#include <cstddef>

namespace gendil {

/** Return a vector map's input size through its `InputSize()` member. */
template<class Map>
GENDIL_HOST_DEVICE
constexpr std::size_t GetInputSize(const Map& map)
   requires requires { map.InputSize(); }
{
   return static_cast<std::size_t>(map.InputSize());
}

/**
 * Return a matrix-like vector map's input size through `num_cols`.
 *
 * This fallback participates only when the preferred `InputSize()` member is
 * unavailable.
 */
template<class Map>
GENDIL_HOST_DEVICE
constexpr std::size_t GetInputSize(const Map& map)
   requires (
      !requires { map.InputSize(); } &&
      requires { map.num_cols; })
{
   return static_cast<std::size_t>(map.num_cols);
}

/** Return a vector map's output size through its `OutputSize()` member. */
template<class Map>
GENDIL_HOST_DEVICE
constexpr std::size_t GetOutputSize(const Map& map)
   requires requires { map.OutputSize(); }
{
   return static_cast<std::size_t>(map.OutputSize());
}

/**
 * Return a matrix-like vector map's output size through `num_rows`.
 *
 * This fallback participates only when the preferred `OutputSize()` member is
 * unavailable.
 */
template<class Map>
GENDIL_HOST_DEVICE
constexpr std::size_t GetOutputSize(const Map& map)
   requires (
      !requires { map.OutputSize(); } &&
      requires { map.num_rows; })
{
   return static_cast<std::size_t>(map.num_rows);
}

/** Apply a callable vector map with replacement semantics. */
template<class Map, class InputVector, class OutputVector>
GENDIL_HOST_DEVICE
void Apply(
   const Map& map,
   const InputVector& input,
   OutputVector& output)
   requires requires
   {
      { map(input, output) } -> std::same_as<void>;
   }
{
   map(input, output);
}

/** Add a vector map's result to the existing output vector. */
template<class Map, class InputVector, class OutputVector>
GENDIL_HOST_DEVICE
void ApplyAdd(
   const Map& map,
   const InputVector& input,
   OutputVector& output)
   requires requires
   {
      { map.ApplyAdd(input, output) } -> std::same_as<void>;
   }
{
   map.ApplyAdd(input, output);
}

/**
 * @brief Require the complete callable vector-map protocol for selected vectors.
 *
 * Replacement and additive application are both part of the protocol. The
 * `operator()` and `Apply` both denote replacement application and must have
 * equivalent semantics. Their semantic equivalence cannot be checked by a
 * C++ concept. ADL-customized maps remain required to be callable.
 */
template<class Map, class InputVector, class OutputVector>
concept VectorMap = requires(
   const Map& map,
   const InputVector& input,
   OutputVector& output)
{
   { map(input, output) } -> std::same_as<void>;
   { GetInputSize(map) } -> std::same_as<std::size_t>;
   { GetOutputSize(map) } -> std::same_as<std::size_t>;
   { Apply(map, input, output) } -> std::same_as<void>;
   { ApplyAdd(map, input, output) } -> std::same_as<void>;
};

} // namespace gendil
