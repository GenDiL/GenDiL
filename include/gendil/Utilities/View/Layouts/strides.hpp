// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Query a tensor layout's strides along each axis.
 *
 * A stride is the change in container index when one tensor coordinate
 * increases by one. These queries inspect strides without depending on how
 * a layout stores them; ordinary tensor access still uses view(i,j).
 *
 * For a 2-by-3 tensor stored with its first coordinate varying fastest:
 * @code
 * #include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
 * auto layout = MakeFIFOStridedLayout(std::array{2, 3});
 * auto step_i = GetStride<0>(layout);  // 1: moving from (i,j) to (i+1,j).
 * auto step_j = GetStride(layout, 1); // 2: moving from (i,j) to (i,j+1).
 * @endcode
 *
 * Axes are zero-based. The template form chooses an axis at compile time;
 * the two-argument form chooses it at runtime. Layouts may support either or
 * both forms. HasStrides lets generic code ask whether every axis supports
 * the template form before using it.
 *
 * Strides are expressed in container entries, not bytes or tensor lengths.
 * They do not establish contiguous access to the underlying values: a
 * container may, for example, apply an indirection after layout indexing.
 */

#include "gendil/Utilities/View/extents.hpp"

#include <type_traits>

namespace gendil {

/**
 * @brief Get the stride along a compile-time-selected axis.
 *
 * Forward to layout.Stride(AxisTag<Axis>{}), preserving its result type.
 * A layout with a type-known stride can therefore return an integral constant;
 * selecting an axis at compile time does not itself make the stride static.
 */
template < size_t Axis, class Layout >
   requires requires( const Layout & layout )
   {
      layout.Stride( AxisTag< Axis >{} );
   }
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetStride( const Layout & layout )
{
   return layout.Stride( AxisTag< Axis >{} );
}

/**
 * @brief Get the stride along an axis selected at runtime.
 *
 * Forward to layout.Stride(axis), preserving its result type. This overload
 * requires that separate member operation; it is not synthesized from the
 * template query. The axis must be valid for the layout; no check is added here.
 */
template < class Layout >
   requires requires( const Layout & layout, const size_t axis )
   {
      layout.Stride( axis );
   }
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetStride(
   const Layout & layout,
   const size_t axis )
{
   return layout.Stride( axis );
}

namespace detail {

/** Check one axis before folding, avoiding a requires-expression over a pack. */
template < class Layout, size_t Axis >
concept HasStrideQuery = requires( const Layout & layout )
{
   GetStride< Axis >( layout );
};

/** Check that the template query is callable for each axis in the sequence. */
template < class Layout, size_t... Axis >
consteval bool HasStridesImpl( std::index_sequence< Axis... > )
{
   return ( HasStrideQuery< Layout, Axis > && ... );
}

} // namespace detail

/**
 * @brief Whether all tensor axes expose a compile-time-selected stride query.
 *
 * After removing cv/ref qualifiers, require a constant rank and a callable
 * GetStride<I> for every axis below it. No runtime-axis query or numeric return
 * type is required, and stride values are not validated. A rank-zero layout
 * needs no queries; types without a suitable rank return false.
 * The unqualified queries also find layout-specific free overloads through
 * argument-dependent lookup.
 */
template < class Layout >
inline constexpr bool has_strides_v =
   []
   {
      if constexpr ( requires
      {
         typename std::integral_constant<
            size_t, std::remove_cvref_t< Layout >::rank >;
      } )
      {
         return detail::HasStridesImpl< std::remove_cvref_t< Layout > >(
            std::make_index_sequence< std::remove_cvref_t< Layout >::rank >{} );
      }
      else { return false; }
   }();

/** Constrain a template to layouts satisfying has_strides_v. */
template < class Layout >
concept HasStrides = has_strides_v< Layout >;

} // namespace gendil
