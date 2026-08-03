// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/Loop/deviceloop.hpp"

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

template < typename OutputValueType, typename ComputeType >
GENDIL_HOST_DEVICE
constexpr void CheckSparseOutputScalingArithmetic()
{
   static_assert(
      std::is_convertible_v< OutputValueType, ComputeType >,
      "Scaled sparse matrix apply requires output vector values to be "
      "convertible to the backend compute type." );
   using ScaledOutputType =
      decltype(
         std::declval< ComputeType >() *
         std::declval< ComputeType >() );
   static_assert(
      std::is_convertible_v< ScaledOutputType, OutputValueType >,
      "Scaled sparse matrix apply requires scaled output values to be "
      "convertible to the output vector scalar type." );
}

template <
   typename OutputValue,
   typename IndexType,
   typename ComputeType >
void ScaleSparseHostOutput(
   OutputValue * output,
   const IndexType size,
   const ComputeType beta )
{
   CheckSparseOutputScalingArithmetic< OutputValue, ComputeType >();
   if ( beta == ComputeType( 1 ) )
   {
      return;
   }

   #pragma omp parallel for
   for ( IndexType index = 0; index < size; ++index )
   {
      if ( beta == ComputeType( 0 ) )
      {
         output[index] = OutputValue( 0 );
      }
      else
      {
         output[index] =
            static_cast< OutputValue >(
               beta * static_cast< ComputeType >( output[index] ) );
      }
   }
}

template <
   typename OutputValue,
   typename IndexType,
   typename ComputeType >
void ScaleSparseDeviceOutput(
   OutputValue * output,
   const IndexType size,
   const ComputeType beta )
{
   static_assert(
      std::is_integral_v< IndexType >,
      "Sparse output dimensions must use an integral type." );
   CheckSparseOutputScalingArithmetic< OutputValue, ComputeType >();

   if constexpr ( std::is_signed_v< IndexType > )
   {
      GENDIL_VERIFY(
         size >= 0,
         "Cannot scale a sparse output with a negative dimension." );
   }
   if ( beta == ComputeType( 1 ) )
   {
      return;
   }

   DeviceLoop(
      size,
      [=] GENDIL_HOST_DEVICE ( const IndexType index )
      {
         if ( beta == ComputeType( 0 ) )
         {
            output[index] = OutputValue( 0 );
         }
         else
         {
            output[index] =
               static_cast< OutputValue >(
                  beta * static_cast< ComputeType >( output[index] ) );
         }
      } );
}

} // namespace gendil
