// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Restriction-backed element tensor views and vector storage binding.
 */

#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionlayout.hpp"
#include "gendil/FiniteElementMethod/Restrictions/weightedrestrictionview.hpp"
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/vectorrestriction.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/tensorview.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"
#include "gendil/Utilities/View/view.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/** @brief Bind algebraic storage to one tensor-leaf restriction. */
template < typename Restriction, typename Container >
   requires (
      TensorElementDoFRestriction< Restriction > &&
      static_restriction_entry_count_v<
         std::remove_cvref_t< Restriction > > == 1 &&
      restriction_supports_element_reference_view_v< Restriction > )
GENDIL_HOST_DEVICE
auto MakeRestrictionElementView(
   const Restriction & restriction,
   const Container & data )
{
   return MakeView( data, MakeRestrictionLayout( restriction ) );
}

/** @brief Bind algebraic storage through weighted local-DoF proxies. */
template < typename Restriction, typename Container >
   requires (
      TensorElementDoFRestriction< Restriction > &&
      !( static_restriction_entry_count_v<
            std::remove_cvref_t< Restriction > > == 1 &&
         restriction_supports_element_reference_view_v< Restriction > ) )
GENDIL_HOST_DEVICE
auto MakeRestrictionElementView(
   const Restriction & restriction,
   const Container & data )
{
   return MakeWeightedRestrictionView( restriction, data );
}

namespace details {

template < typename Restriction, typename Container, size_t... Component >
   requires VectorElementDoFRestriction< Restriction >
GENDIL_HOST_DEVICE
auto MakeVectorRestrictionElementView(
   const Restriction & restriction,
   const Container & data,
   std::index_sequence< Component... > )
{
   return std::make_tuple(
      MakeRestrictionElementView(
         GetComponentRestriction< Component >( restriction ),
         data )... );
}

} // namespace details

/** @brief Bind one algebraic vector to a compile-time tuple of component views. */
template < typename Restriction, typename Container >
   requires VectorElementDoFRestriction< Restriction >
GENDIL_HOST_DEVICE
auto MakeRestrictionElementView(
   const Restriction & restriction,
   const Container & data )
{
   using RestrictionType = std::remove_cvref_t< Restriction >;
   return details::MakeVectorRestrictionElementView(
      restriction,
      data,
      std::make_index_sequence<
         static_cast< size_t >( RestrictionType::num_components ) >{} );
}

/** @brief Compatibility name for constructing a scalar element tensor view. */
template < typename FiniteElementSpace, typename Container >
   requires TensorElementDoFRestriction<
      typename std::remove_cvref_t<
         FiniteElementSpace >::restriction_type >
auto MakeScalarElementTensorView(
   const FiniteElementSpace & finite_element_space,
   const Container & data )
{
   return MakeRestrictionElementView(
      GetRestriction( finite_element_space ),
      data );
}

/** @brief Compatibility overload selecting explicit vector components. */
template <
   typename FiniteElementSpace,
   typename Container,
   size_t... Component >
   requires VectorElementDoFRestriction<
      typename std::remove_cvref_t<
         FiniteElementSpace >::restriction_type >
auto MakeVectorElementTensorView(
   const FiniteElementSpace & finite_element_space,
   const Container & data,
   std::index_sequence< Component... > components )
{
   return details::MakeVectorRestrictionElementView(
      GetRestriction( finite_element_space ),
      data,
      components );
}

/** @brief Compatibility name for constructing all vector component views. */
template < typename FiniteElementSpace, typename Container >
   requires VectorElementDoFRestriction<
      typename std::remove_cvref_t<
         FiniteElementSpace >::restriction_type >
auto MakeVectorElementTensorView(
   const FiniteElementSpace & finite_element_space,
   const Container & data )
{
   return MakeRestrictionElementView(
      GetRestriction( finite_element_space ),
      data );
}

/** @brief Construct scalar or vector element views from the stored restriction. */
template < typename FiniteElementSpace, typename Container >
auto MakeElementTensorView(
   const FiniteElementSpace & finite_element_space,
   const Container & data )
{
   return MakeRestrictionElementView(
      GetRestriction( finite_element_space ),
      data );
}

/** @brief Element-view type produced for a space and scalar storage type. */
template < typename FiniteElementSpace, typename T >
using element_tensor_view_t = decltype(
   MakeElementTensorView(
      std::declval< const FiniteElementSpace & >(),
      std::declval< T * >() ) );

/** @brief Bind read-only storage in the kernel policy's memory space. */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename VectorType >
   requires KernelReadableVector<
      is_device_configuration_v< KernelPolicy >,
      VectorType >
auto MakeReadOnlyElementTensorView(
   const FiniteElementSpace & finite_element_space,
   const VectorType & data )
{
   static_assert(
      is_host_configuration_v< KernelPolicy > ||
         is_device_configuration_v< KernelPolicy >,
      "Element tensor views require a host or device kernel policy." );
   constexpr bool on_device =
      is_device_configuration_v< KernelPolicy >;
   auto kernel_data = ReadKernelVector< on_device >( data );
   return MakeElementTensorView(
      finite_element_space,
      kernel_data.data );
}

/** @brief Bind write-only storage in the kernel policy's memory space. */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename VectorType >
   requires KernelWritableVector<
      is_device_configuration_v< KernelPolicy >,
      VectorType >
auto MakeWriteOnlyElementTensorView(
   const FiniteElementSpace & finite_element_space,
   VectorType & data )
{
   static_assert(
      is_host_configuration_v< KernelPolicy > ||
         is_device_configuration_v< KernelPolicy >,
      "Element tensor views require a host or device kernel policy." );
   constexpr bool on_device =
      is_device_configuration_v< KernelPolicy >;
   auto kernel_data = WriteKernelVector< on_device >( data );
   return MakeElementTensorView(
      finite_element_space,
      kernel_data.data );
}

/** @brief Bind read-write storage in the kernel policy's memory space. */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename VectorType >
   requires KernelWritableVector<
      is_device_configuration_v< KernelPolicy >,
      VectorType >
auto MakeReadWriteElementTensorView(
   const FiniteElementSpace & finite_element_space,
   VectorType & data )
{
   static_assert(
      is_host_configuration_v< KernelPolicy > ||
         is_device_configuration_v< KernelPolicy >,
      "Element tensor views require a host or device kernel policy." );
   constexpr bool on_device =
      is_device_configuration_v< KernelPolicy >;
   auto kernel_data = ReadWriteKernelVector< on_device >( data );
   return MakeElementTensorView(
      finite_element_space,
      kernel_data.data );
}

} // namespace gendil
