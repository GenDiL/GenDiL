// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/KernelContext/threadlayout.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"

namespace gendil
{

template < typename KernelConfig >
struct is_serial
{
   static constexpr bool value = is_serial<typename KernelConfig::thread_layout_type>::value;
};

template <>
struct is_serial< ThreadBlockLayout<> >
{
   static constexpr bool value = true;
};

template < size_t FirstDim, size_t ... RestDims >
struct is_serial< ThreadBlockLayout< FirstDim, RestDims ... > >
{
   static constexpr bool value = false;
};

template < typename KernelConfiguration, Integer SharedBlockSize >
struct is_serial< KernelContext< KernelConfiguration, SharedBlockSize > >
{
   static constexpr bool value = is_serial<typename KernelConfiguration::thread_layout_type>::value;
};

template < typename KernelConfig >
static constexpr bool is_serial_v = is_serial< KernelConfig >::value;

} // namespace gendil
