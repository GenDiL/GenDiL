// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <array>
#include <tuple>
#include <utility>

#include "gendil/Meshes/Connectivities/Orientations/tensorproductorientation.hpp"

namespace gendil {

namespace detail
{

template < size_t Offset, Integer Dim, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
void NativeToReferenceIndexInto(
   const std::array< Integer, Dim > & native,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Rank > & orientation,
   std::array< Integer, Dim > & reference )
{
   static_assert( Offset + Rank <= static_cast< size_t >( Dim ) );
   gendil::ConstexprLoop< Rank >( [&] ( auto native_axis )
   {
      const LocalIndex o = orientation( native_axis );
      const size_t reference_axis =
         static_cast< size_t >( o > 0 ? o - 1 : -o - 1 );
      reference[ Offset + reference_axis ] = o > 0
         ? native[ Offset + native_axis ]
         : static_cast< Integer >(
              sizes[ Offset + native_axis ] - 1 -
              native[ Offset + native_axis ] );
   } );
}

template < size_t Offset, Integer Dim, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
void NativeToReferenceIndexInto(
   const std::array< Integer, Dim > & native,
   const std::array< size_t, Dim > &,
   const IdentityOrientation< Rank > &,
   std::array< Integer, Dim > & reference )
{
   static_assert( Offset + Rank <= static_cast< size_t >( Dim ) );
   gendil::ConstexprLoop< Rank >( [&] ( auto axis )
   {
      reference[ Offset + axis ] = native[ Offset + axis ];
   } );
}

template < size_t Offset, Integer Dim, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
void NativeToReferenceIndexInto(
   const std::array< Integer, Dim > & native,
   const std::array< size_t, Dim > & sizes,
   const TensorProductOrientation< Orientations... > & orientation,
   std::array< Integer, Dim > & reference )
{
   if constexpr ( is_runtime_permutation_orientation_v<
      TensorProductOrientation< Orientations... > > )
   {
      constexpr size_t ProductDim =
         TensorProductOrientation< Orientations... >::Dim;
      for ( size_t local_native_axis = 0;
            local_native_axis < ProductDim;
            ++local_native_axis )
      {
         const size_t native_axis = Offset + local_native_axis;
         const LocalIndex mapped_axis =
            GetRuntimeStructuredOrientationAxis< Offset >(
               native_axis,
               orientation );
         const size_t reference_axis = static_cast< size_t >(
            mapped_axis > 0 ? mapped_axis - 1 : -mapped_axis - 1 );
         reference[ reference_axis ] = mapped_axis > 0
            ? native[ native_axis ]
            : static_cast< Integer >(
                 sizes[ native_axis ] - 1 - native[ native_axis ] );
      }
   }
   else
   {
      using OrientationTuple = std::tuple< Orientations... >;
      gendil::ConstexprLoop< sizeof...( Orientations ) >( [&] ( auto component )
      {
         constexpr size_t ComponentOffset = Offset +
            orientation_component_offset_v< component, OrientationTuple >;
         NativeToReferenceIndexInto< ComponentOffset >(
            native,
            sizes,
            orientation.template Get< component >(),
            reference );
      } );
   }
}

} // namespace detail

/** @brief Convert a flat-oriented native index to reference axis order. */
template < Integer Dim >
GENDIL_HOST_DEVICE GENDIL_INLINE
std::array< Integer, Dim > NativeToReferenceIndex(
   const std::array< Integer, Dim > & native,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   std::array< Integer, Dim > reference{};

   for ( size_t native_axis = 0; native_axis < Dim; ++native_axis )
   {
      const int o = orientation( native_axis );
      const size_t reference_axis =
         static_cast< size_t >( o > 0 ? o - 1 : -o - 1 );

      reference[ reference_axis ] = o > 0
         ? native[ native_axis ]
         : static_cast< Integer >(
              sizes[ native_axis ] - 1 - native[ native_axis ] );
   }

   return reference;
}

/** @brief Convert through a static identity orientation without runtime work. */
template < Integer Dim, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr std::array< Integer, Dim > NativeToReferenceIndex(
   const std::array< Integer, Dim > & native,
   const std::array< size_t, Dim > &,
   const IdentityOrientation< Rank > & )
{
   static_assert(
      Rank == static_cast< size_t >( Dim ),
      "Identity orientation and index dimensions must match." );
   return native;
}

/** @brief Convert a structured native index directly to reference axis order. */
template < Integer Dim, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
std::array< Integer, Dim > NativeToReferenceIndex(
   const std::array< Integer, Dim > & native,
   const std::array< size_t, Dim > & sizes,
   const TensorProductOrientation< Orientations... > & orientation )
{
   static_assert(
      TensorProductOrientation< Orientations... >::Dim ==
         static_cast< size_t >( Dim ),
      "Tensor-product orientation and index dimensions must match." );
   std::array< Integer, Dim > reference{};
   detail::NativeToReferenceIndexInto< 0 >(
      native,
      sizes,
      orientation,
      reference );
   return reference;
}

template < Integer Dim, size_t... Is >
GENDIL_HOST_DEVICE GENDIL_INLINE
std::array< Integer, Dim > NativeToReferenceIndex(
   const std::array< Integer, Dim > & native,
   const std::index_sequence< Is... > & sizes,
   const Permutation< Dim > & orientation )
{
   return NativeToReferenceIndex(
      native,
      std::array< size_t, Dim >{ Is... },
      orientation );
}

template < Integer Dim, size_t... Is, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
constexpr std::array< Integer, Dim > NativeToReferenceIndex(
   const std::array< Integer, Dim > & native,
   const std::index_sequence< Is... > &,
   const IdentityOrientation< Rank > & orientation )
{
   static_assert( sizeof...( Is ) == static_cast< size_t >( Dim ) );
   return NativeToReferenceIndex(
      native,
      std::array< size_t, Dim >{ Is... },
      orientation );
}

template < Integer Dim, size_t... Is, class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
std::array< Integer, Dim > NativeToReferenceIndex(
   const std::array< Integer, Dim > & native,
   const std::index_sequence< Is... > &,
   const TensorProductOrientation< Orientations... > & orientation )
{
   static_assert( sizeof...( Is ) == static_cast< size_t >( Dim ) );
   return NativeToReferenceIndex(
      native,
      std::array< size_t, Dim >{ Is... },
      orientation );
}

} // namespace gendil
