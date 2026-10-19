// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/staticalgebraconcepts.hpp"

namespace gendil::mapping_detail
{

// Mapping updates Real-valued static vectors in place, so readable entries
// must also accept Real assignments.
template<class V>
concept MappingVector =
   StaticVector<V, Real> &&
   std::same_as<vector_value_t<V>, Real> &&
   requires(V& vector) { GetValue(vector, Integer{}) = Real{}; };

} // namespace gendil::mapping_detail
