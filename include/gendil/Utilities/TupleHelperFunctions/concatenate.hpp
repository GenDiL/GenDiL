// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{
namespace tuple_detail
{
template<class Left, class Right, std::size_t... I, std::size_t... J>
GENDIL_HOST_DEVICE
constexpr auto ConcatenateImpl(Left&& left, Right&& right,
   std::index_sequence<I...>, std::index_sequence<J...>)
{
   using Result = std::tuple<
      std::tuple_element_t<I, std::remove_cvref_t<Left>>...,
      std::tuple_element_t<J, std::remove_cvref_t<Right>>...>;
   return Result{std::get<I>(std::forward<Left>(left))...,
                 std::get<J>(std::forward<Right>(right))...};
}
} // namespace tuple_detail

// Expression construction can return non-literal objects. NVCC's relaxed
// constexpr support does not make std::tuple_cat device-callable for those
// instantiations; this helper explicitly supports both host and device.

/// Concatenate two tuples, preserving declared element types and reference members.
template<class Left, class Right>
GENDIL_HOST_DEVICE
constexpr auto Concatenate(Left&& left, Right&& right)
{
   return tuple_detail::ConcatenateImpl(std::forward<Left>(left), std::forward<Right>(right),
      std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Left>>>{},
      std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Right>>>{});
}
} // namespace gendil
