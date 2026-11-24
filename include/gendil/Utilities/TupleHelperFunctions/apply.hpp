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
template<class Fn, class Tuple, std::size_t... I>
GENDIL_HOST_DEVICE
constexpr decltype(auto) ApplyImpl(Fn&& fn, Tuple&& tuple, std::index_sequence<I...>)
{
   return std::forward<Fn>(fn)(std::get<I>(std::forward<Tuple>(tuple))...);
}
} // namespace tuple_detail

// Expression construction can return non-literal objects. NVCC's relaxed
// constexpr support does not make std::apply device-callable for those
// instantiations; this helper explicitly supports both host and device.

/// Invoke a directly callable function object with forwarded tuple elements.
/// Preserves the return type, including references; does not invoke member pointers.
template<class Fn, class Tuple>
   requires requires { std::tuple_size<std::remove_cvref_t<Tuple>>::value; }
GENDIL_HOST_DEVICE
constexpr decltype(auto) Apply(Fn&& fn, Tuple&& tuple)
{
   return tuple_detail::ApplyImpl(std::forward<Fn>(fn), std::forward<Tuple>(tuple),
      std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Tuple>>>{});
}
} // namespace gendil
