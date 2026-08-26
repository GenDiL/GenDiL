// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "gendil/Utilities/types.hpp"

namespace gendil
{

/**
 * @brief Strongly typed data composed as a tensor product.
 *
 * Entries may themselves be tensor-product data. The interpretation of each
 * entry is supplied by the consuming subsystem.
 */
template<class... Entries>
struct TensorProductData
{
   static_assert(
      sizeof...(Entries) > 0,
      "TensorProductData requires at least one entry.");

   using entries_type = std::tuple<Entries...>;
   entries_type entries;
};

template<class... Entries>
GENDIL_HOST_DEVICE
constexpr auto MakeTensorProductData(Entries&&... entries)
{
   using Data = TensorProductData<std::remove_cvref_t<Entries>...>;
   using Tuple = typename Data::entries_type;
   return Data{
      Tuple{std::forward<Entries>(entries)...}};
}

template<size_t I, class... Entries>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetTensorProductEntry(
   TensorProductData<Entries...>& data)
{
   return std::get<I>(data.entries);
}

template<size_t I, class... Entries>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetTensorProductEntry(
   const TensorProductData<Entries...>& data)
{
   return std::get<I>(data.entries);
}

template<size_t I, class... Entries>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetTensorProductEntry(
   TensorProductData<Entries...>&& data)
{
   return std::get<I>(std::move(data.entries));
}

template<size_t I, class... Entries>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetTensorProductEntry(
   const TensorProductData<Entries...>&& data)
{
   return std::get<I>(std::move(data.entries));
}

template<class T>
struct is_tensor_product_data : std::false_type
{
};

template<class... Entries>
struct is_tensor_product_data<TensorProductData<Entries...>>
   : std::true_type
{
};

template<class T>
inline constexpr bool is_tensor_product_data_v =
   is_tensor_product_data<std::remove_cvref_t<T>>::value;

template<size_t I, class Data>
using tensor_product_entry_t =
   std::tuple_element_t<
      I,
      typename std::remove_cvref_t<Data>::entries_type>;

template<class Data>
inline constexpr size_t tensor_product_entry_count_v =
   std::tuple_size_v<
      typename std::remove_cvref_t<Data>::entries_type>;

template<class Tuple>
struct tensor_product_data_from_tuple;

template<class... Entries>
struct tensor_product_data_from_tuple<std::tuple<Entries...>>
{
   using type = TensorProductData<Entries...>;
};

template<class Tuple>
using tensor_product_data_from_tuple_t =
   typename tensor_product_data_from_tuple<
      std::remove_cvref_t<Tuple>>::type;

} // namespace gendil
