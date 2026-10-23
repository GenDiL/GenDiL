// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "gendil/Meshes/Connectivities/Orientations/identityorientation.hpp"
#include "gendil/Utilities/Loop/loops.hpp"

namespace gendil {

namespace detail
{

template < size_t I, class Orientation, bool = std::is_empty_v< Orientation > >
struct TensorProductOrientationComponentStorage;

template < size_t I, class Orientation >
struct TensorProductOrientationComponentStorage< I, Orientation, false >
{
   Orientation orientation;

   GENDIL_HOST_DEVICE
   constexpr TensorProductOrientationComponentStorage() = default;

   GENDIL_HOST_DEVICE
   constexpr explicit TensorProductOrientationComponentStorage(
      const Orientation & orientation_ )
      : orientation( orientation_ )
   {}

   GENDIL_HOST_DEVICE GENDIL_INLINE
   constexpr const Orientation & Get() const
   {
      return orientation;
   }
};

template < size_t I, class Orientation >
struct TensorProductOrientationComponentStorage< I, Orientation, true >
{
   GENDIL_HOST_DEVICE
   constexpr TensorProductOrientationComponentStorage() = default;

   GENDIL_HOST_DEVICE
   constexpr explicit TensorProductOrientationComponentStorage(
      const Orientation & )
   {}

   GENDIL_HOST_DEVICE GENDIL_INLINE
   constexpr Orientation Get() const
   {
      return {};
   }
};

template < class Indices, class... Orientations >
struct TensorProductOrientationStorage;

template < size_t... I, class... Orientations >
struct TensorProductOrientationStorage<
   std::index_sequence< I... >,
   Orientations... >
   : TensorProductOrientationComponentStorage< I, Orientations >...
{
   GENDIL_HOST_DEVICE
   constexpr TensorProductOrientationStorage() = default;

   GENDIL_HOST_DEVICE
   constexpr explicit TensorProductOrientationStorage(
      const Orientations &... orientations )
      : TensorProductOrientationComponentStorage< I, Orientations >(
           orientations )...
   {}

   template < size_t Component >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   constexpr decltype(auto) Get() const
   {
      using Orientation =
         std::tuple_element_t< Component, std::tuple< Orientations... > >;
      using Storage =
         TensorProductOrientationComponentStorage< Component, Orientation >;
      return static_cast< const Storage & >( *this ).Get();
   }
};

template < size_t Component, class OrientationTuple, size_t... I >
consteval size_t OrientationComponentOffsetImpl( std::index_sequence< I... > )
{
   return ( size_t{ 0 } + ... +
      orientation_dimension_v< std::tuple_element_t< I, OrientationTuple > > );
}

template < size_t Component, class OrientationTuple >
inline constexpr size_t orientation_component_offset_v =
   OrientationComponentOffsetImpl< Component, OrientationTuple >(
      std::make_index_sequence< Component >{} );

} // namespace detail

/**
 * @brief Orientation whose components preserve tensor-product cell blocks.
 *
 * Each component orientation acts only on the matching ProductCell factor.
 * Runtime permutations may therefore coexist with statically encoded identity
 * factors without materializing a full-product permutation.
 */
template < class... Orientations >
struct TensorProductOrientation
   : detail::TensorProductOrientationStorage<
        std::index_sequence_for< Orientations... >,
        Orientations... >
{
   static_assert(
      sizeof...( Orientations ) > 0,
      "TensorProductOrientation requires at least one component." );

   using component_types = std::tuple< Orientations... >;
   using base = detail::TensorProductOrientationStorage<
      std::index_sequence_for< Orientations... >,
      Orientations... >;

   static constexpr size_t num_components = sizeof...( Orientations );
   static constexpr size_t Dim =
      ( size_t{ 0 } + ... + orientation_dimension_v< Orientations > );

   using base::base;
   using base::Get;
};

template < class... Orientations >
TensorProductOrientation( const Orientations &... )
   -> TensorProductOrientation< std::remove_cvref_t< Orientations >... >;

template < class... Orientations >
struct OrientationDimension< TensorProductOrientation< Orientations... > >
   : std::integral_constant<
        size_t,
        ( size_t{ 0 } + ... + orientation_dimension_v< Orientations > ) >
{};

template < class Orientation >
struct IsTensorProductOrientation : std::false_type
{};

template < class... Orientations >
struct IsTensorProductOrientation<
   TensorProductOrientation< Orientations... > > : std::true_type
{};

template < class Orientation >
inline constexpr bool is_tensor_product_orientation_v =
   IsTensorProductOrientation< std::remove_cvref_t< Orientation > >::value;

template < class... Orientations >
struct IsRuntimePermutationOrientation<
   TensorProductOrientation< Orientations... > >
   : std::bool_constant<
        ( IsRuntimePermutationOrientation< Orientations >::value && ... ) >
{};

template < class... Orientations >
struct IsStaticIdentityOrientation<
   TensorProductOrientation< Orientations... > >
   : std::bool_constant<
        ( is_static_identity_orientation_v< Orientations > && ... ) >
{};

/** @brief Validate each independent factor of a tensor-product orientation. */
template < class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr bool IsValidOrientation(
   const TensorProductOrientation< Orientations... > & orientation )
{
   return [&]< size_t... Component >( std::index_sequence< Component... > )
   {
      return ( IsValidOrientation( orientation.template Get< Component >() ) && ... );
   }( std::index_sequence_for< Orientations... >{} );
}

/**
 * @brief Construct a compact factor-preserving product orientation.
 *
 * Empty static factors use base-class elision and contribute no runtime
 * payload. If every factor is static identity, the result is the single
 * full-dimensional IdentityOrientation type.
 */
template < class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto MakeTensorProductOrientation(
   const Orientations &... orientations )
{
   constexpr size_t Dim =
      ( size_t{ 0 } + ... + orientation_dimension_v< Orientations > );
   if constexpr ( ( is_static_identity_orientation_v< Orientations > && ... ) )
   {
      return IdentityOrientation< Dim >{};
   }
   else
   {
      return TensorProductOrientation<
         std::remove_cvref_t< Orientations >... >( orientations... );
   }
}

namespace detail
{

/** @brief Read a runtime structured axis at a global native-axis index. */
template < size_t Offset, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr LocalIndex GetRuntimeStructuredOrientationAxis(
   const size_t native_axis,
   const Permutation< Rank > & orientation )
{
   const LocalIndex local_axis = orientation( native_axis - Offset );
   return OffsetOrientationAxis( local_axis, Offset );
}

template < size_t Offset, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr LocalIndex GetRuntimeStructuredOrientationAxis(
   size_t native_axis,
   const TensorProductOrientation< Orientations... > & orientation );

template <
   size_t Offset,
   size_t Component,
   class ProductOrientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr LocalIndex GetTensorProductRuntimeOrientationAxis(
   const size_t native_axis,
   const ProductOrientation & orientation )
{
   static_assert( Component < ProductOrientation::num_components );
   using OrientationTuple = typename ProductOrientation::component_types;
   using ComponentOrientation =
      std::tuple_element_t< Component, OrientationTuple >;
   constexpr size_t ComponentOffset = Offset +
      orientation_component_offset_v< Component, OrientationTuple >;
   constexpr size_t ComponentEnd = ComponentOffset +
      orientation_dimension_v< ComponentOrientation >;
   if constexpr ( Component + 1 == ProductOrientation::num_components )
   {
      return GetRuntimeStructuredOrientationAxis< ComponentOffset >(
         native_axis,
         orientation.template Get< Component >() );
   }
   else
   {
      if ( native_axis < ComponentEnd )
      {
         return GetRuntimeStructuredOrientationAxis< ComponentOffset >(
            native_axis,
            orientation.template Get< Component >() );
      }
      else
      {
         return GetTensorProductRuntimeOrientationAxis<
            Offset,
            Component + 1 >( native_axis, orientation );
      }
   }
}

/**
 * @brief Read an axis from a product containing runtime permutations only.
 *
 * This runtime lookup is used by the linear all-runtime layout and index
 * transformations without flattening the product orientation.
 */
template < size_t Offset, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr LocalIndex GetRuntimeStructuredOrientationAxis(
   const size_t native_axis,
   const TensorProductOrientation< Orientations... > & orientation )
{
   static_assert(
      is_runtime_permutation_orientation_v< decltype( orientation ) > );
   return GetTensorProductRuntimeOrientationAxis< Offset, 0 >(
      native_axis,
      orientation );
}

/** @brief Read a selected runtime axis within its tensor-product factor. */
template < size_t Axis, size_t Offset = 0, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr LocalIndex GetStructuredOrientationAxis(
   const Permutation< Rank > & orientation )
{
   static_assert( Offset <= Axis && Axis < Offset + Rank );
   return OffsetOrientationAxis( orientation( Axis - Offset ), Offset );
}

/**
 * @brief Preserve a static identity axis without runtime orientation storage.
 *
 * Nonidentity static permutations are deliberately unsupported: consumers of
 * static axes rely on identity ordering. Use `Permutation` for other mappings.
 */
template < size_t Axis, size_t Offset = 0, size_t Rank, Permutation< Rank > Value >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto GetStructuredOrientationAxis(
   const std::integral_constant< Permutation< Rank >, Value > & )
{
   static_assert( Offset <= Axis && Axis < Offset + Rank );
   static_assert( Value == MakeReferencePermutation< Rank >(),
      "Only static identity orientations are supported; use Permutation for signed or reordered axes." );
   return std::integral_constant< LocalIndex, Axis + 1 >{};
}

template < size_t Axis, size_t Offset = 0, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto GetStructuredOrientationAxis(
   const TensorProductOrientation< Orientations... > & orientation );

template < size_t Axis, size_t Offset, size_t Component, class Product >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto GetTensorProductOrientationAxis( const Product & orientation )
{
   static_assert( Component < Product::num_components );
   using ComponentType = std::tuple_element_t< Component,
                                              typename Product::component_types >;
   constexpr size_t component_offset = Offset +
      orientation_component_offset_v< Component, typename Product::component_types >;
   if constexpr ( Axis < component_offset + orientation_dimension_v< ComponentType > )
      return GetStructuredOrientationAxis< Axis, component_offset >(
         orientation.template Get< Component >() );
   else
      return GetTensorProductOrientationAxis< Axis, Offset, Component + 1 >( orientation );
}

/** Read a signed axis without flattening a structured orientation. */
template < size_t Axis, size_t Offset, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto GetStructuredOrientationAxis(
   const TensorProductOrientation< Orientations... > & orientation )
{
   static_assert( Offset <= Axis &&
      Axis < Offset + TensorProductOrientation< Orientations... >::Dim );
   return GetTensorProductOrientationAxis< Axis, Offset, 0 >( orientation );
}

/** @brief Whether a structured-axis result is known at compile time. */
template < class Axis >
inline constexpr bool is_static_orientation_axis_v = requires
{
   std::remove_cvref_t< Axis >::value;
};

template < class ProductOrientation, size_t... I >
GENDIL_HOST_DEVICE
constexpr auto FlattenTensorProductOrientation(
   const ProductOrientation & orientation,
   std::index_sequence< I... > )
{
   constexpr size_t Dim = orientation_dimension_v< ProductOrientation >;
   using OrientationTuple = typename ProductOrientation::component_types;
   auto result = MakeReferencePermutation< Dim >();
   ( Set< orientation_component_offset_v< I, OrientationTuple > >(
        result,
        FlattenOrientation( orientation.template Get< I >() ) ), ... );
   return result;
}

} // namespace detail

/**
 * @brief Materialize a tensor-product orientation as one flat permutation.
 *
 * This operation is provided for diagnostics, equivalence testing, and
 * external interoperability. Production geometry, DoF, layout, and sparse
 * paths consume `TensorProductOrientation` directly.
 */
template < class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr auto FlattenOrientation(
   const TensorProductOrientation< Orientations... > & orientation )
{
   return detail::FlattenTensorProductOrientation(
      orientation,
      std::index_sequence_for< Orientations... >{} );
}

} // namespace gendil
