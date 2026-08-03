// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/vendorsparsecommon.hpp"
#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/cusparsebackend.hpp"
#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/rocsparsebackend.hpp"

namespace gendil
{

#if defined(GENDIL_USE_CUDA)
template < typename ComputeType = void >
using VendorDeviceBSRBackend = CuSparseBSRBackend< ComputeType >;
template < typename ComputeType = void >
using VendorDeviceCOOBackend = CuSparseCOOBackend< ComputeType >;
template < typename ComputeType = void >
using VendorDeviceCSRBackend = CuSparseCSRBackend< ComputeType >;
template < typename ComputeType = void >
using VendorDeviceCSCBackend = CuSparseCSCBackend< ComputeType >;
#elif defined(GENDIL_USE_HIP)
template < typename ComputeType = void >
using VendorDeviceBSRBackend = RocSparseBSRBackend< ComputeType >;
template < typename ComputeType = void >
using VendorDeviceCOOBackend = RocSparseCOOBackend< ComputeType >;
template < typename ComputeType = void >
using VendorDeviceCSRBackend = RocSparseCSRBackend< ComputeType >;
template < typename ComputeType = void >
using VendorDeviceCSCBackend = RocSparseCSCBackend< ComputeType >;
#endif

} // namespace gendil
