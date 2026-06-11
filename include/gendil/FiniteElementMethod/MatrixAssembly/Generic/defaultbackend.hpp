// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/bsrmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/coomatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/cscmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/csrmatrix.hpp"
#ifdef GENDIL_USE_HYPRE
#include "gendil/Interfaces/Hypre/hypretypes.hpp"
#endif
#include "gendil/FiniteElementMethod/MatrixAssembly/matrixassemblytype.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/types.hpp"

namespace gendil {

template < MatrixAssemblyType Type >
struct DefaultBackendFor
{
   static_assert(
      dependent_false_value_v< Type >,
      "DefaultBackendFor is defined only for implemented assembly formats. "
      "The requested matrix assembly format is not implemented yet." );
};

template <>
struct DefaultBackendFor< MatrixAssemblyType::BSR >
{
#if defined(GENDIL_USE_DEVICE)
   using type = NativeDeviceBSRBackend<>;
#else
   using type = HostBSRBackend<>;
#endif
};

template <>
struct DefaultBackendFor< MatrixAssemblyType::SGBSR >
{
#if defined(GENDIL_USE_DEVICE)
   using type = NativeDeviceBSRBackend<>;
#else
   using type = HostBSRBackend<>;
#endif
};

template <>
struct DefaultBackendFor< MatrixAssemblyType::RawCOO >
{
   using type = Empty;
};

template <>
struct DefaultBackendFor< MatrixAssemblyType::COO >
{
#if defined(GENDIL_USE_DEVICE)
   using type = NativeDeviceCOOBackend<>;
#else
   using type = HostCOOBackend<>;
#endif
};

template <>
struct DefaultBackendFor< MatrixAssemblyType::CSR >
{
#if defined(GENDIL_USE_DEVICE)
   using type = NativeDeviceCSRBackend<>;
#else
   using type = HostCSRBackend<>;
#endif
};

#ifdef GENDIL_USE_HYPRE
template <>
struct DefaultBackendFor< MatrixAssemblyType::HypreCSR >
{
#ifdef GENDIL_USE_HYPRE_DEVICE
   using type = HypreCSRDeviceBackend;
#else
   using type = HypreCSRHostBackend;
#endif
};
#endif

template <>
struct DefaultBackendFor< MatrixAssemblyType::CSC >
{
#if defined(GENDIL_USE_DEVICE)
   using type = NativeDeviceCSCBackend<>;
#else
   using type = HostCSCBackend<>;
#endif
};

template < MatrixAssemblyType Type >
using DefaultBackendFor_t = typename DefaultBackendFor< Type >::type;

} // namespace gendil
