// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file orienteddofvalidation.hpp
 * @brief Compatibility checks for tensor-product DoF shapes and orientations.
 *
 * The constexpr query functions perform validation when called. The kernel
 * verifier retains compile-time dimension checking everywhere, while its
 * runtime connectivity checks exist only in host debug compilation.
 */

#include <array>
#include <tuple>
#include <type_traits>

#include "gendil/Meshes/Connectivities/Orientations/tensorproductorientation.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/toarray.hpp"
#include "gendil/Utilities/types.hpp"

namespace gendil {

namespace detail {

template < size_t Offset, Integer Dim, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool OrientationExtentsAreCompatibleAt(
   const std::array< size_t, Dim > & sizes,
   const Permutation< Rank > & orientation )
{
   static_assert( Offset + Rank <= static_cast< size_t >( Dim ) );
   for ( size_t native_axis = 0; native_axis < Rank; ++native_axis )
   {
      const LocalIndex mapped_axis = orientation( native_axis );
      const size_t reference_axis = static_cast< size_t >(
         mapped_axis > 0 ? mapped_axis - 1 : -mapped_axis - 1 );
      if ( reference_axis >= Rank ||
           sizes[ Offset + native_axis ] !=
              sizes[ Offset + reference_axis ] )
      {
         return false;
      }
   }
   return true;
}

template < size_t Offset, Integer Dim, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool OrientationExtentsAreCompatibleAt(
   const std::array< size_t, Dim > &,
   const IdentityOrientation< Rank > & )
{
   static_assert( Offset + Rank <= static_cast< size_t >( Dim ) );
   return true;
}

template < size_t Offset, Integer Dim, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool OrientationExtentsAreCompatibleAt(
   const std::array< size_t, Dim > & sizes,
   const TensorProductOrientation< Orientations... > & orientation )
{
   using OrientationTuple = std::tuple< Orientations... >;
   bool compatible = true;
   gendil::ConstexprLoop< sizeof...( Orientations ) >( [&] ( auto component )
   {
      constexpr size_t ComponentOffset = Offset +
         orientation_component_offset_v< component, OrientationTuple >;
      compatible = compatible &&
         OrientationExtentsAreCompatibleAt< ComponentOffset >(
            sizes,
            orientation.template Get< component >() );
   } );
   return compatible;
}

} // namespace detail

/**
 * @brief Return whether a flat orientation preserves a tensor DoF shape.
 *
 * Axis reversals are always compatible. Axis permutations are compatible only
 * when the corresponding native and reference extents are equal.
 */
template < Integer Dim >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool OrientedTensorDofExtentsAreCompatible(
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   for ( Integer native_axis = 0; native_axis < Dim; ++native_axis )
   {
      const LocalIndex mapped_axis = orientation( native_axis );
      const Integer reference_axis = static_cast< Integer >(
         mapped_axis > 0 ? mapped_axis - 1 : -( mapped_axis + 1 ) );
      if ( reference_axis >= Dim ||
           sizes[ native_axis ] != sizes[ reference_axis ] )
      {
         return false;
      }
   }
   return true;
}

/** @brief Return whether a structured orientation preserves a tensor shape. */
template < Integer Dim, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool OrientedTensorDofExtentsAreCompatible(
   const std::array< size_t, Dim > & sizes,
   const TensorProductOrientation< Orientations... > & orientation )
{
   static_assert(
      orientation_dimension_v< decltype( orientation ) > ==
         static_cast< size_t >( Dim ) );
   return detail::OrientationExtentsAreCompatibleAt< 0 >(
      sizes,
      orientation );
}

/** @brief Static identity preserves every matching-dimensional tensor shape. */
template < Integer Dim >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool OrientedTensorDofExtentsAreCompatible(
   const std::array< size_t, Dim > &,
   const IdentityOrientation< Dim > & )
{
   return true;
}

/** @brief Return whether an orientation is valid and shape-compatible. */
template < Integer Dim, class Orientation >
requires (
   orientation_dimension_v< Orientation > == static_cast< size_t >( Dim ) )
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool OrientedTensorDofShapeIsCompatible(
   const std::array< size_t, Dim > & sizes,
   const Orientation & orientation )
{
   return IsValidOrientation( orientation ) &&
      OrientedTensorDofExtentsAreCompatible( sizes, orientation );
}

/**
 * @brief Verify that an orientation can index a compile-time tensor DoF shape.
 *
 * The dimension assertion is always instantiated. Runtime permutation and
 * extent checks are host-debug diagnostics only and produce no Release or
 * device instructions.
 */
template < typename DofShape, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
void VerifyOrientedTensorDofShapeCompatibility(
   [[maybe_unused]] const Orientation & orientation )
{
   constexpr Integer Dim = static_cast< Integer >(
      orientation_dimension_v< Orientation > );
   static_assert(
      DofShape::size() == Dim,
      "Mismatching oriented tensor-product DOF shape and orientation dimensions." );

#if !defined(GENDIL_DEVICE_CODE) && !defined(NDEBUG)
   const auto sizes = to_array( DofShape{} );
   const bool valid = IsValidOrientation( orientation );
   GENDIL_ASSERT(
      valid,
      "DOF orientation must be a signed permutation." );
   if ( valid )
   {
      GENDIL_ASSERT(
         OrientedTensorDofExtentsAreCompatible( sizes, orientation ),
         "DOF orientation permutes local tensor axes with incompatible extents." );
   }
#endif
}

} // namespace gendil
