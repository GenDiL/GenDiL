// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/KernelConfiguration/threadlayout.hpp"
#include "gendil/Utilities/KernelConfiguration/kernelcontext.hpp"

namespace gendil
{

template < size_t I, typename KernelConfig >
struct is_threaded_dim;

template < size_t I >
struct is_threaded_dim< I, ThreadBlockLayout<> >
{
   static constexpr bool value = false;
};

template < size_t I, size_t ... Dims >
struct is_threaded_dim< I, ThreadBlockLayout< Dims ... > >
{
   // Note: This assumes the first dimensions are threaded
   static constexpr bool value = I < sizeof...(Dims);
};

template < size_t I, size_t NumSharedDimensions, typename ThreadLayout, typename FESpace, typename IntegrationRule >
struct is_threaded_dim< I, KernelContext< NumSharedDimensions, ThreadLayout, FESpace, IntegrationRule > >
{
   static constexpr bool value = is_threaded_dim<I, ThreadLayout>::value;
};

template < typename KernelConfig >
static constexpr bool is_threaded_dim_v = is_threaded_dim< KernelConfig >::value;

} // namespace gendil
