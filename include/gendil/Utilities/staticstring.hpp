// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "types.hpp"

#include <cstddef>
#include <string_view>
#include <algorithm>

namespace gendil
{

template <std::size_t N>
struct StaticString
{
   char chars[N]{};

   constexpr StaticString(const char (&s)[N])
   {
      std::copy_n(s, N, chars);
   }

   constexpr std::string_view view() const
   {
      return std::string_view(chars, N-1);
   }
};

template <std::size_t A, std::size_t B>
constexpr bool operator==(const StaticString<A>& lhs, const StaticString<B>& rhs) noexcept
{
  // Fast path: lengths differ (excluding null terminator)
  if constexpr (A != B) {
    // But A and B include '\0'. If they differ, the string lengths differ -> not equal.
    return false;
  } else {
    for (std::size_t i = 0; i < A; ++i) {
      if (lhs.chars[i] != rhs.chars[i]) return false;
    }
    return true;
  }
}

template <std::size_t A, std::size_t B>
constexpr bool operator!=(const StaticString<A>& lhs, const StaticString<B>& rhs) noexcept
{
  return !(lhs == rhs);
}

template <std::size_t N>
StaticString(const char(&)[N]) -> StaticString<N>;

template <std::size_t N>
std::ostream& operator<<(std::ostream& os, const StaticString<N>& string)
{
   return os << string.view();
}

template <std::size_t firstN, std::size_t... RestN>
std::ostream& operator<<(std::ostream& os, const std::tuple< StaticString<firstN>, StaticString<RestN>... >& string)
{  
   std::apply([&](auto const&... xs) {
      std::size_t i = 0;
      ((os << (i++ ? ", " : "") << xs), ...);
   }, string);
   return os;
}

} // namespace gendil
