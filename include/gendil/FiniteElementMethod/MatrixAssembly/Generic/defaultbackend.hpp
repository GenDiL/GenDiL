// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/bsrmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/coomatrix.hpp"
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
      "CSR and CSC assembly are not implemented yet." );
};

template <>
struct DefaultBackendFor< MatrixAssemblyType::BSR >
{
#if defined(GENDIL_USE_DEVICE)
   using type = NativeDeviceBSRBackend;
#else
   using type = HostBSRBackend;
#endif
};

template <>
struct DefaultBackendFor< MatrixAssemblyType::SGBSR >
{
#if defined(GENDIL_USE_DEVICE)
   using type = NativeDeviceBSRBackend;
#else
   using type = HostBSRBackend;
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
   using type = NativeDeviceCOOBackend;
#else
   using type = HostCOOBackend;
#endif
};

template < MatrixAssemblyType Type >
using DefaultBackendFor_t = typename DefaultBackendFor< Type >::type;

} // namespace gendil
