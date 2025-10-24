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

template < typename IntegrationRule, size_t ... extra_dims >
struct QuadraturePointValues;

template < Integer Index, typename IntegrationRule, size_t ... extra_dims >
struct GetTensorSize< Index, QuadraturePointValues< IntegrationRule, extra_dims ... > >
: GetTensorSize< Index, typename QuadraturePointValues< IntegrationRule, extra_dims ... >::Data > {};

template < Integer Index, typename Container, typename Sizes, Integer ... Strides >
struct GetTensorSize< Index, FixedStridedView< Container, Sizes, Strides ... > >
{
   static constexpr size_t value = seq_get_v< Index, Sizes >;
};

template < typename Sizes, typename KernelContext, typename Container >
struct ThreadedView;

template < Integer Index, typename Sizes, typename KernelContext, typename Container >
struct GetTensorSize< Index, ThreadedView< Sizes, KernelContext, Container > >
{
   // !FIXME Is it really what we want?
   static constexpr size_t value = GetTensorSize< Index, Container >::value;
};

template < Integer Index, typename RecursiveArrayType >
constexpr Integer get_tensor_size_v = GetTensorSize< Index, RecursiveArrayType >::value;

} // namespace gendil
