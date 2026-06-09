// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

template < typename ValueType, typename InputValueType >
using DefaultSparseComputeType_t =
   std::common_type_t< ValueType, InputValueType >;

template < typename ComputeType >
using DefaultSparseAccumulatorType_t = ComputeType;

template <
   typename Backend,
   typename ValueType,
   typename InputValueType >
using ResolveSparseComputeType_t =
   std::conditional_t<
      std::is_void_v< typename Backend::compute_type >,
      DefaultSparseComputeType_t< ValueType, InputValueType >,
      typename Backend::compute_type >;

template <
   typename Backend,
   typename ComputeType >
using ResolveSparseAccumulatorType_t =
   std::conditional_t<
      std::is_void_v< typename Backend::accumulator_type >,
      DefaultSparseAccumulatorType_t< ComputeType >,
      typename Backend::accumulator_type >;

template < typename Pointer >
using SparsePointerValueType_t =
   std::remove_cv_t< std::remove_pointer_t< Pointer > >;

template <
   typename ValueType,
   typename InputValueType,
   typename OutputValueType,
   typename ComputeType,
   typename AccumulatorType >
GENDIL_HOST_DEVICE
constexpr void CheckRowOwnedSparseApplyArithmetic()
{
   static_assert(
      std::is_convertible_v< ValueType, ComputeType >,
      "Sparse matrix Apply requires matrix values to be convertible to "
      "the backend compute type." );
   static_assert(
      std::is_convertible_v< InputValueType, ComputeType >,
      "Sparse matrix Apply requires input vector values to be convertible "
      "to the backend compute type." );

   using ProductType =
      decltype(
         std::declval< ComputeType >() *
         std::declval< ComputeType >() );
   static_assert(
      std::is_convertible_v< ProductType, AccumulatorType >,
      "Row-owned sparse matrix Apply requires computed products to be "
      "convertible to the backend accumulator type." );
   static_assert(
      std::is_convertible_v< AccumulatorType, OutputValueType >,
      "Row-owned sparse matrix Apply requires the accumulator type to be "
      "convertible to the output vector scalar type." );
}

template <
   typename ValueType,
   typename InputValueType,
   typename OutputValueType,
   typename ComputeType >
GENDIL_HOST_DEVICE
constexpr void CheckScatterSparseApplyArithmetic()
{
   static_assert(
      std::is_convertible_v< ValueType, ComputeType >,
      "Sparse matrix Apply requires matrix values to be convertible to "
      "the backend compute type." );
   static_assert(
      std::is_convertible_v< InputValueType, ComputeType >,
      "Sparse matrix Apply requires input vector values to be convertible "
      "to the backend compute type." );

   using ProductType =
      decltype(
         std::declval< ComputeType >() *
         std::declval< ComputeType >() );
   static_assert(
      std::is_convertible_v< ProductType, OutputValueType >,
      "Scatter sparse matrix Apply requires computed contributions to be "
      "convertible to the output vector scalar type." );
}

} // namespace gendil
