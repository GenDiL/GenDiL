// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace gendil {

namespace internal {

template < Integer N >
struct PowHelper
{
   template < typename T >
   GENDIL_HOST_DEVICE
   static constexpr T eval( T const & val )
   {
      return val * PowHelper< N -1 >::eval( val );
   }
};

template < >
struct PowHelper< 1 >
{
   template < typename T >
   GENDIL_HOST_DEVICE
   static constexpr T eval( T const & val )
   {
      return val;
   }
};

template <  >
struct PowHelper< 0 >
{
   template < typename T >
   GENDIL_HOST_DEVICE
   static constexpr T eval( T const & val )
   {
      return 1;
   }
};

}

/**
 * @brief Computes the power of a value.
 * 
 * @tparam power Exponent of the exponentiation.
 * @tparam T Type of the input value.
 * @param val The input value.
 * @return constexpr auto The result of the exponentiation.
 */
template < Integer power, typename T >
GENDIL_HOST_DEVICE
constexpr auto Pow( const T & val )
{
   return internal::PowHelper< power >::eval( val );
}

}
