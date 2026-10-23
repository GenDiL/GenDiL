// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/get.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"

namespace gendil
{

template < Integer Index, typename RecursiveArrayType >
struct GetTensorSize;

template < Integer Index, typename Container, typename Layout >
   requires (
      has_extent_v<
         static_cast< size_t >( Index ),
         View< Container, Layout > > &&
      detail::is_integral_constant_v< decltype(
         GetExtent< static_cast< size_t >( Index ) >(
            std::declval< const View< Container, Layout > & >() ) ) > )
struct GetTensorSize< Index, View< Container, Layout > >
{
   static constexpr size_t value = static_cast< size_t >(
      std::remove_cvref_t< decltype( GetExtent< static_cast< size_t >( Index ) >(
         std::declval< const View< Container, Layout > & >() ) ) >::value );
};

template < typename IntegrationRule, size_t ... extra_dims >
struct QuadraturePointValues;

template < Integer Index, typename IntegrationRule, size_t ... extra_dims >
struct GetTensorSize< Index, QuadraturePointValues< IntegrationRule, extra_dims ... > >
: GetTensorSize< Index, typename QuadraturePointValues< IntegrationRule, extra_dims ... >::Data > {};

template < typename Sizes, typename KernelContext, typename Container >
struct ThreadedView;

template < Integer Index, typename Sizes, typename KernelContext, typename Container >
struct GetTensorSize< Index, ThreadedView< Sizes, KernelContext, Container > >
{
   static constexpr size_t value = seq_get_v< Index, Sizes >;
};

template < Integer Index, typename RecursiveArrayType >
constexpr Integer get_tensor_size_v = GetTensorSize< Index, RecursiveArrayType >::value;

} // namespace gendil
