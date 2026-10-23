// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

namespace gendil
{

// Retain the vector's storage and scalar type. Readability alone does not
// guarantee that a new writable result of that type can be returned.
template < typename T, StaticVector<T> VecType >
   requires std::same_as<T, vector_value_t<VecType>> &&
            (!std::is_array_v<VecType>) &&
            std::default_initializable<VecType> &&
            std::move_constructible<VecType> &&
            requires(VecType& out, const VecType& in, const T& scalar, GlobalIndex i) {
               GetValue(out, i) = scalar * static_cast<T>(GetValue(in, i));
            }
GENDIL_HOST_DEVICE
constexpr auto operator*( const T & scalar, const VecType & vec )
{
   VecType result;
   ConstexprLoop< static_extent_v<VecType> >( [&] ( auto i )
   {
      GetValue(result, i) = scalar * static_cast<T>(GetValue(vec, i));
   });
   return result;
}

} // namespace gendil
