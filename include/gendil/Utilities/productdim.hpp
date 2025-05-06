// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil
{

template <typename... Ts>
struct product_dim;

template <typename T>
struct product_dim<T>
{
   static constexpr Integer value = std::decay_t<T>::Dim;
};

template <typename T, typename... Ts>
struct product_dim<T, Ts...>
{
   static constexpr Integer value = std::decay_t<T>::Dim + product_dim<Ts...>::value;
};

template <typename... Ts>
struct product_dim<std::tuple<Ts...>> : product_dim<Ts...>
{ };

/// @brief Gets the sum of the dimensions of types in a product structure, e.g. product cell/mesh/etc...
template <typename... Meshes>
constexpr Integer product_dim_v = product_dim<Meshes...>::value;

} // namespace gendil
