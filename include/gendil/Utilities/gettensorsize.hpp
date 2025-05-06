// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil
{

template < Integer Index, typename RecursiveArrayType >
struct GetTensorSize;

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
