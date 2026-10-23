// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Runtime row-orientation signs applied to completed restrictions.
 *
 * `SignedRestriction<Restriction>` decorates a completed restriction with one
 * Boolean sign per element-local restriction row. A false flag represents
 * @f$+1@f$ and a true flag represents @f$-1@f$. If the wrapped restriction is
 * @f$E@f$ and the row signs form the diagonal matrix @f$D@f$, the adaptor
 * represents @f$DE@f$; adjoint scattering consequently applies
 * @f$E^*D@f$.
 *
 * The sign map is borrowed. Scalar tensor rows use element-major, flattened
 * local-DoF order; a wrapped tensor-product restriction therefore retains its
 * existing product-row order. Vector rows concatenate those scalar row ranges
 * in component-major order. A row sign multiplies every entry of that row;
 * entry-specific coefficients remain weights of the wrapped restriction.
 */

#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionweight.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/vectorrestriction.hpp"
#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionconcepts.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictiontraits.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionvalidation.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

#include <array>
#include <type_traits>
#include <utility>

namespace gendil {

namespace details {

template < typename Restriction, bool IsVector >
struct SignedRestrictionMetadata;

template < typename Restriction >
struct SignedRestrictionMetadata< Restriction, false >
{
   static_assert(
      TensorElementDoFRestriction< Restriction >,
      "A scalar SignedRestriction requires a tensor-shaped restriction." );

   using dof_shape_type = typename Restriction::dof_shape_type;
   static constexpr Integer tensor_dim = Restriction::tensor_dim;
};

template < typename Restriction >
struct SignedRestrictionMetadata< Restriction, true >
{
   static constexpr Integer num_components = Restriction::num_components;
};

/** @brief Return a shallow sign-map handle advanced by a row offset. */
GENDIL_HOST_DEVICE
inline HostDevicePointer< const bool > OffsetRestrictionSignMap(
   const HostDevicePointer< const bool > signs,
   const GlobalIndex offset )
{
   HostDevicePointer< const bool > result = signs;
   if ( result.host_pointer != nullptr )
   {
      result.host_pointer += offset;
   }
#ifdef GENDIL_USE_DEVICE
   if ( result.device_pointer != nullptr )
   {
      result.device_pointer += offset;
   }
#endif
   return result;
}

/** @brief Sum vector-child row counts preceding one component. */
template < size_t Component, typename Restriction, size_t... Previous >
GENDIL_HOST_DEVICE
constexpr GlobalIndex SignedRestrictionComponentRowOffset(
   const Restriction & restriction,
   std::index_sequence< Previous... > )
{
   static_assert(
      VectorElementDoFRestriction< Restriction >,
      "Signed restriction component offsets require a vector restriction." );
   return ( GlobalIndex{ 0 } + ... +
      GetNumberOfLocalDofs(
         GetComponentRestriction< Previous >( restriction ) ) );
}

} // namespace details

/**
 * @brief Completed restriction decorated by one runtime sign per local row.
 *
 * The wrapped restriction and sign handle are stored by value; pointer-backed
 * maps remain borrowed. Cardinality and algebraic coordinates are unchanged.
 */
template < ElementDoFRestriction Restriction >
struct SignedRestriction
   : details::SignedRestrictionMetadata<
        Restriction,
        VectorElementDoFRestriction< Restriction > >
{
   using restriction_type = Restriction;

   Restriction restriction;
   HostDevicePointer< const bool > row_is_negative;
   GlobalIndex sign_count;

   GENDIL_HOST_DEVICE
   constexpr SignedRestriction(
      const Restriction & restriction_,
      const HostDevicePointer< const bool > row_is_negative_,
      const GlobalIndex sign_count_ )
      : restriction( restriction_ ),
        row_is_negative( row_is_negative_ ),
        sign_count( sign_count_ )
   { }
};

/** @brief Select a signed child and its component-major sign-map slice. */
template < size_t Component, typename Restriction >
   requires VectorElementDoFRestriction< Restriction >
GENDIL_HOST_DEVICE
constexpr auto GetComponentRestriction(
   const SignedRestriction< Restriction > & restriction )
{
   static_assert(
      Component < static_cast< size_t >( Restriction::num_components ),
      "Signed vector restriction component index is out of bounds." );
   const auto & child =
      GetComponentRestriction< Component >( restriction.restriction );
   using Child = std::remove_cvref_t< decltype( child ) >;
   const GlobalIndex row_offset =
      details::SignedRestrictionComponentRowOffset< Component >(
         restriction.restriction,
         std::make_index_sequence< Component >{} );
   const GlobalIndex child_rows = GetNumberOfLocalDofs( child );
   return SignedRestriction< Child >{
      child,
      details::OffsetRestrictionSignMap(
         restriction.row_is_negative,
         row_offset ),
      child_rows };
}

/** @brief Expose the signed component type of a signed vector restriction. */
template < typename Restriction, size_t Component >
   requires VectorElementDoFRestriction< Restriction >
struct RestrictionComponent<
   SignedRestriction< Restriction >,
   Component >
{
   static_assert(
      Component < static_cast< size_t >( Restriction::num_components ),
      "Signed vector restriction component index is out of bounds." );
   using type = SignedRestriction<
      restriction_component_t< Restriction, Component > >;
};

/** @brief Visit a signed scalar tensor row and compose its runtime sign. */
template < typename Restriction, typename Visitor >
   requires TensorElementDoFRestriction< Restriction >
GENDIL_HOST_DEVICE
constexpr void ForEachRestrictionEntry(
   const SignedRestriction< Restriction > & restriction,
   const GlobalIndex element,
   const std::array< GlobalIndex, Restriction::tensor_dim > & local_dof,
   Visitor && visitor )
{
   using DofShape = typename Restriction::dof_shape_type;
   constexpr GlobalIndex element_rows = Product( DofShape{} );
   GENDIL_ASSERT(
      details::DofIndexIsInBounds< DofShape >( local_dof ),
      "SignedRestriction local tensor index is out of bounds." );
   const GlobalIndex row =
      element * element_rows + FlattenMultiIndex< DofShape >( local_dof );
   GENDIL_ASSERT(
      row < restriction.sign_count,
      "SignedRestriction local row exceeds its sign map." );

   const RestrictionSign row_sign{
      restriction.row_is_negative[row] };
   auto visitor_value = std::forward< Visitor >( visitor );
   ForEachRestrictionEntry(
      restriction.restriction,
      element,
      local_dof,
      [&] ( const GlobalIndex algebraic_dof, const auto & entry_weight )
      {
         visitor_value(
            algebraic_dof,
            ComposeRestrictionWeights( row_sign, entry_weight ) );
      } );
}

/** @brief Visit a signed vector row through its selected signed child. */
template <
   size_t Component,
   Integer Dim,
   typename Restriction,
   typename Visitor >
   requires VectorElementDoFRestriction< Restriction >
GENDIL_HOST_DEVICE
constexpr void ForEachRestrictionEntry(
   const SignedRestriction< Restriction > & restriction,
   const GlobalIndex element,
   const LocalComponentDoFIndex< Component, Dim > & local_dof,
   Visitor && visitor )
{
   const auto child = GetComponentRestriction< Component >( restriction );
   ForEachRestrictionEntry(
      child,
      element,
      local_dof.local_dof,
      std::forward< Visitor >( visitor ) );
}

/** @brief Return the wrapped restriction's local-row count. */
template < ElementDoFRestriction Restriction >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfLocalDofs(
   const SignedRestriction< Restriction > & restriction )
{
   return GetNumberOfLocalDofs( restriction.restriction );
}

/** @brief Return the wrapped restriction's logical global DoF count. */
template < ElementDoFRestriction Restriction >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetNumberOfGlobalDofs(
   const SignedRestriction< Restriction > & restriction )
{
   return GetNumberOfGlobalDofs( restriction.restriction );
}

/** @brief Return the wrapped restriction's algebraic coordinate extent. */
template < ElementDoFRestriction Restriction >
GENDIL_HOST_DEVICE
constexpr GlobalIndex GetAlgebraicDofExtent(
   const SignedRestriction< Restriction > & restriction )
{
   return GetAlgebraicDofExtent( restriction.restriction );
}

template < ElementDoFRestriction Restriction >
inline constexpr size_t static_restriction_entry_count_v<
   SignedRestriction< Restriction > > =
      static_restriction_entry_count_v< Restriction >;

template <
   ElementDoFRestriction Restriction,
   typename NativeRowTag >
inline constexpr size_t static_restriction_entry_count_for_v<
   SignedRestriction< Restriction >,
   NativeRowTag > =
      static_restriction_entry_count_for_v< Restriction, NativeRowTag >;

template < ElementDoFRestriction Restriction >
inline constexpr bool restriction_may_share_global_dofs_v<
   SignedRestriction< Restriction > > =
      restriction_may_share_global_dofs_v< Restriction >;

template < ElementDoFRestriction Restriction >
inline constexpr bool restriction_rows_have_unique_coordinates_v<
   SignedRestriction< Restriction > > =
      restriction_rows_have_unique_coordinates_v< Restriction >;

template < ElementDoFRestriction Restriction >
struct restriction_supports_element_reference_view<
   SignedRestriction< Restriction > > : std::false_type { };

/** @brief Validate the wrapped map and runtime sign-map residency. */
template < bool OnDevice, ElementDoFRestriction Restriction >
void ValidateRestrictionMemoryAccess(
   const SignedRestriction< Restriction > & restriction,
   const GlobalIndex active_row_count )
{
   ValidateRestrictionMemoryAccess< OnDevice >(
      restriction.restriction,
      active_row_count );
   if constexpr ( OnDevice )
   {
#ifdef GENDIL_USE_DEVICE
      GENDIL_VERIFY(
         active_row_count == 0 ||
            restriction.row_is_negative.device_pointer != nullptr,
         "Device signed-restriction access requires a device-resident sign map." );
#else
      static_assert(
         !OnDevice,
         "Device signed-restriction access requires GenDiL device support." );
#endif
   }
   else
   {
      GENDIL_VERIFY(
         active_row_count == 0 ||
            restriction.row_is_negative.host_pointer != nullptr,
         "Host signed-restriction access requires a host-resident sign map." );
   }
}

/** @brief Validate the wrapped representation and exact sign-map length. */
template < ElementDoFRestriction Restriction >
void ValidateRestrictionRepresentation(
   const SignedRestriction< Restriction > & restriction )
{
   ValidateElementDoFRestriction( restriction.restriction );
   GENDIL_VERIFY(
      restriction.sign_count ==
         GetNumberOfLocalDofs( restriction.restriction ),
      "SignedRestriction requires exactly one sign per local restriction row." );
}

/**
 * @brief Decorate a completed restriction with a borrowed runtime sign map.
 *
 * @param restriction Completed restriction whose coordinates and entry
 * weights are retained.
 * @param row_is_negative Borrowed Boolean flags in local-row order.
 * @param sign_count Number of available flags; it must equal the wrapped
 * restriction's local-row count.
 */
template < ElementDoFRestriction Restriction >
auto MakeSignedRestriction(
   const Restriction & restriction,
   const HostDevicePointer< const bool > row_is_negative,
   const GlobalIndex sign_count )
{
   SignedRestriction< Restriction > result{
      restriction,
      row_is_negative,
      sign_count };
   ValidateElementDoFRestriction( result );
   return result;
}

} // namespace gendil
