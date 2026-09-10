// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief BLAS-like operations on vector-access objects. */

#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/Utilities/Loop/kernelloop.hpp"

namespace gendil {

/**
 * @brief Zero every vector entry in the selected kernel memory space.
 *
 * This operation acquires write-only storage because it overwrites the complete
 * vector. The selected memory space becomes valid and the opposite space is
 * invalidated according to the vector access contract.
 */
template<bool OnDevice, class VectorType>
   requires KernelWritableVector<OnDevice, VectorType>
void Zero(VectorType& vector)
{
   auto data = WriteKernelVector<OnDevice>(vector);
   KernelLoop<OnDevice>(
      GetVectorSize(vector),
      [data] GENDIL_HOST_DEVICE (const size_t i) mutable
      {
         data[i] = Real{0};
      });
}

/** @brief Copy between equally sized vector-access objects. */
template<bool OnDevice, class InputVector, class OutputVector>
   requires KernelReadableVector<OnDevice, InputVector> &&
            KernelWritableVector<OnDevice, OutputVector>
void CopyVector(const InputVector& input, OutputVector& output)
{
   GENDIL_VERIFY(
      GetVectorSize(input) == GetVectorSize(output),
      "Vector copy requires matching extents." );
   auto source = ReadKernelVector<OnDevice>(input);
   auto destination = WriteKernelVector<OnDevice>(output);
   KernelLoop<OnDevice>(GetVectorSize(input), [=] GENDIL_HOST_DEVICE (size_t i) mutable
   {
      destination[i] = source[i];
   });
}

/** @brief Scale any mutable vector-access object in place. */
template<bool OnDevice, class VectorType>
   requires KernelWritableVector<OnDevice, VectorType>
void ScaleVector(const Real coefficient, VectorType& vector)
{
   auto data = ReadWriteKernelVector<OnDevice>(vector);
   KernelLoop<OnDevice>(GetVectorSize(vector), [=] GENDIL_HOST_DEVICE (size_t i) mutable
   {
      data[i] *= coefficient;
   });
}

/** @brief Perform y += a*x for equally sized vector-access objects. */
template<bool OnDevice, class InputVector, class OutputVector>
   requires KernelReadableVector<OnDevice, InputVector> &&
            KernelWritableVector<OnDevice, OutputVector>
void AxpyVector(
   const Real coefficient,
   const InputVector& input,
   OutputVector& output)
{
   GENDIL_VERIFY(
      GetVectorSize(input) == GetVectorSize(output),
      "AXPY requires matching vector extents." );
   auto source = ReadKernelVector<OnDevice>(input);
   auto destination = ReadWriteKernelVector<OnDevice>(output);
   KernelLoop<OnDevice>(GetVectorSize(input), [=] GENDIL_HOST_DEVICE (size_t i) mutable
   {
      destination[i] += coefficient * source[i];
   });
}

} // namespace gendil
