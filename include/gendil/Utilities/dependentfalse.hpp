// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil
{

// Helper for delayed static_assert in template contexts.
// Usage: static_assert(dependent_false_v<T>, "message");
// This allows static_assert to trigger only when the template is instantiated,
// rather than at parse time, which is necessary for SFINAE and template specialization.
template<class...>
inline constexpr bool dependent_false_v = false;

template<auto...>
inline constexpr bool dependent_false_value_v = false;

} // namespace gendil
