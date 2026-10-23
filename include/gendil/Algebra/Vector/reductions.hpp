// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Compute vector products, robust lengths, and basis combinations.
 *
 * EuclideanDot binds caller-owned scratch to a host-driven kernel policy.
 * ScaledEuclideanNorm and ProjectEuclidean explicitly select robust and batched
 * Euclidean arithmetic. For a different inner product, use NormFromDot and
 * ProjectFromDot with the actual callable; no metric customization is inferred.
 */

#include "gendil/Algebra/Vector/localoperations.hpp"
#include "gendil/Algebra/Vector/operations.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"
#include "gendil/Utilities/MathHelperFunctions/isfinite.hpp"
#include "gendil/Utilities/MathHelperFunctions/checkedsqrt.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace gendil {
namespace vector_reduction_detail {

inline constexpr size_t ReductionChunkSize = 256;

inline size_t ChunkCount(const size_t size)
{
   return size / ReductionChunkSize + (size % ReductionChunkSize != 0);
}

inline size_t PartCount(const size_t size, const size_t capacity)
{
   return std::max(size_t{1}, std::min(capacity, ChunkCount(size)));
}

inline size_t CheckedScratchSize(const size_t rows, const size_t columns)
{
   GENDIL_VERIFY(rows == 0 || columns <= std::numeric_limits<size_t>::max() / rows,
      "Vector reduction scratch size overflow.");
   return rows * columns;
}

// Only the active scalar/coefficient prefix crosses the host/device boundary.
inline void CopyResultsToHost(Real* host, Real* device, const size_t count)
{
#ifdef GENDIL_USE_DEVICE
   GENDIL_VERIFY(count <= static_cast<size_t>(std::numeric_limits<GlobalIndex>::max()),
      "Reduction result transfer exceeds the supported index range.");
   HostDevicePointer<Real> transfer;
   transfer.host_pointer = host;
   transfer.device_pointer = device;
   ToHost(static_cast<GlobalIndex>(count), transfer);
#else
   (void)host;
   (void)device;
   (void)count;
#endif
}

inline void CopyCoefficientsToDevice(Real* host, Real* device, const size_t count)
{
#ifdef GENDIL_USE_DEVICE
   GENDIL_VERIFY(count <= static_cast<size_t>(std::numeric_limits<GlobalIndex>::max()),
      "Coefficient transfer exceeds the supported index range.");
   HostDevicePointer<Real> transfer;
   transfer.host_pointer = host;
   transfer.device_pointer = device;
   ToDevice(static_cast<GlobalIndex>(count), transfer);
#else
   (void)host;
   (void)device;
   (void)count;
#endif
}

GENDIL_HOST_DEVICE inline Real Magnitude(const Real value)
{
   return value < 0.0 ? -value : value;
}

// Represent sum(value^2) as scale^2 * sum without squaring unscaled entries.
struct ScaledSquares
{
   Real scale = 0.0;
   Real sum = 0.0;

   GENDIL_HOST_DEVICE void Add(const Real value)
   {
      if (!IsFinite(value))
      {
         scale = std::numeric_limits<Real>::quiet_NaN();
         sum = scale;
         return;
      }
      const Real magnitude = Magnitude(value);
      if (magnitude == 0.0) { return; }
      if (scale < magnitude)
      {
         const Real ratio = scale / magnitude;
         sum = 1.0 + sum * ratio * ratio;
         scale = magnitude;
      }
      else
      {
         const Real ratio = magnitude / scale;
         sum += ratio * ratio;
      }
   }

   GENDIL_HOST_DEVICE void Merge(const ScaledSquares other)
   {
      if (!IsFinite(other.scale) || !IsFinite(other.sum))
      {
         scale = std::numeric_limits<Real>::quiet_NaN();
         sum = scale;
         return;
      }
      if (other.scale == 0.0) { return; }
      if (scale < other.scale)
      {
         const Real ratio = scale / other.scale;
         sum = other.sum + sum * ratio * ratio;
         scale = other.scale;
      }
      else
      {
         const Real ratio = other.scale / scale;
         sum += other.sum * ratio * ratio;
      }
   }

   GENDIL_HOST_DEVICE Real Norm() const
   {
      return scale == 0.0 ? 0.0 : scale * std::sqrt(sum);
   }
};

template<class Basis>
void ValidateContiguousBasis(Basis& basis, const size_t columns, const size_t size)
{
   if (columns == 0 || size == 0) { return; }
   const size_t stride = basis.ContiguousStride();
   const size_t storage_size = GetVectorSize(basis.ContiguousStorage());
   GENDIL_VERIFY(stride >= size && storage_size >= size &&
      columns - 1 <= (storage_size - size) / stride,
      "Contiguous basis storage does not cover the requested vector extents.");
}

} // namespace vector_reduction_detail

/**
 * @brief Size scratch for a fully partitioned device Euclidean dot product.
 *
 * Counts Real entries, not bytes. Any nonempty device reduction also works with
 * a smaller scratch vector of at least one entry, using fewer partitions.
 * Host policies and empty vectors need no scratch.
 */
inline size_t RequiredDotReductionWorkspaceSize(const size_t vector_size)
{
   return vector_size == 0 ? 0 : 1 + vector_reduction_detail::ChunkCount(vector_size);
}

/// Full parallel scaled-norm capacity; a nonempty device norm needs at least one entry.
inline size_t RequiredNormReductionWorkspaceSize(const size_t vector_size)
{
   return vector_size == 0 ? 0 : 1 + vector_reduction_detail::CheckedScratchSize(
      vector_reduction_detail::ChunkCount(vector_size), size_t{2});
}

/**
 * @brief Size scratch for all projection columns and their parallel partial sums.
 *
 * Smaller scratch is supported down to one entry for nonempty inputs: the
 * implementation reduces fewer partitions or completes each Euclidean dot in
 * turn. Coefficients are host-addressable and have at least count entries.
 */
inline size_t RequiredProjectionReductionWorkspaceSize(
   const size_t vector_size, const size_t count)
{
   return vector_size == 0 ? 0 : vector_reduction_detail::CheckedScratchSize(
      1 + vector_reduction_detail::ChunkCount(vector_size), count);
}

/// Device fused combination transfers count coefficients through this many Real entries.
inline size_t RequiredCombinationWorkspaceSize(const size_t count) { return count; }

/**
 * @brief Compute a Euclidean product without moving full device vectors to the host.
 *
 * The operands have equal extents. Scratch is borrowed and must not alias either
 * operand. A device policy requires at least one scratch entry for nonempty
 * inputs; RequiredDotReductionWorkspaceSize enables full partitioning. Host
 * policies accept empty scratch. The completed scalar is returned on the host.
 */
template<class KernelPolicy, class First, class Second>
   requires KernelReadableVector<is_device_configuration_v<KernelPolicy>, First> &&
            KernelReadableVector<is_device_configuration_v<KernelPolicy>, Second>
Real Dot(Vector& scratch, const First& first, const Second& second)
{
   GENDIL_VERIFY(GetVectorSize(first) == GetVectorSize(second),
      "Euclidean inner product requires matching vector extents.");
   const size_t size = GetVectorSize(first);
   if (size == 0) { return 0.0; }
   constexpr bool device = is_device_configuration_v<KernelPolicy>;
   auto left = ReadKernelVector<device>(first);
   auto right = ReadKernelVector<device>(second);
   if constexpr (!device)
   {
      Real result = 0.0;
      #pragma omp parallel for reduction(+:result)
      for (size_t i = 0; i < size; ++i) { result += left[i] * right[i]; }
      return result;
   }
   else
   {
      GENDIL_VERIFY(scratch.Size() >= 1,
         "A nonempty device dot requires at least one reduction scratch entry.");
      const bool serial = scratch.Size() == 1;
      const size_t parts = serial ? 1 :
         vector_reduction_detail::PartCount(size, scratch.Size() - 1);
      Real* host = scratch.WriteHostData();
      Real* result = scratch.WriteDeviceData();
      Real* partial = result + (serial ? 0 : 1);
      KernelLoop<true>(parts, [=] GENDIL_HOST_DEVICE (const size_t part) mutable
      {
         Real sum = 0.0;
         for (size_t i = part; i < size; i += parts) { sum += left[i] * right[i]; }
         partial[part] = sum;
      });
      if (!serial)
      {
         KernelLoop<true>(size_t{1}, [=] GENDIL_HOST_DEVICE (size_t) mutable
         {
            Real sum = 0.0;
            for (size_t p = 0; p < parts; ++p) { sum += partial[p]; }
            result[0] = sum;
         });
      }
      vector_reduction_detail::CopyResultsToHost(host, result, 1);
      return host[0];
   }
}

/**
 * @brief Bind reusable scratch to an explicitly Euclidean inner product.
 *
 * @code
 * Vector scratch(RequiredNormReductionWorkspaceSize(x.Size()));
 * EuclideanDot<HostKernelConfiguration> dot{scratch};
 * Real product = dot(x, y);
 * Real length = ScaledEuclideanNorm(dot, x);
 * @endcode
 *
 * This object borrows scratch by reference; it neither owns nor prepares solver
 * storage. Keep scratch alive and do not reuse it concurrently. Dot's scratch
 * and operand requirements apply to each call.
 */
template<class KernelPolicy>
struct EuclideanDot
{
   Vector& reduction_scratch;

   template<class First, class Second>
   Real operator()(const First& first, const Second& second) const
   {
      return Dot<KernelPolicy>(reduction_scratch, first, second);
   }
};

/** Induced norm through the exact supplied dot; an overflowing self-dot yields NaN. */
template<class DotFunction, class Input>
GENDIL_HOST_DEVICE Real NormFromDot(DotFunction&& dot, const Input& input)
{
   return CheckedSqrt(dot(input, input));
}

/**
 * @brief Measure Euclidean length without squaring unscaled vector entries.
 *
 * Finite representable norms survive large or tiny entries. Nonfinite entries
 * produce NaN. Scratch follows Dot's minimum requirement; the norm sizing
 * helper provides the two-state partition capacity. Existing Norml2 behavior
 * is unchanged.
 */
template<class KernelPolicy, class Input>
Real ScaledEuclideanNorm(const EuclideanDot<KernelPolicy>& dot, const Input& input)
{
   auto& scratch = dot.reduction_scratch;
   const size_t size = GetVectorSize(input);
   if (size == 0) { return 0.0; }
   constexpr bool device = is_device_configuration_v<KernelPolicy>;
   auto data = ReadKernelVector<device>(input);
   if constexpr (!device)
   {
      Real scale = 0.0;
      int invalid = 0;
      #pragma omp parallel for reduction(max:scale) reduction(|:invalid)
      for (size_t i = 0; i < size; ++i)
      {
         const Real value = data[i];
         invalid |= !IsFinite(value);
         scale = std::max(scale, vector_reduction_detail::Magnitude(value));
      }
      if (invalid) { return std::numeric_limits<Real>::quiet_NaN(); }
      if (scale == 0.0) { return 0.0; }
      Real sum = 0.0;
      #pragma omp parallel for reduction(+:sum)
      for (size_t i = 0; i < size; ++i)
      {
         const Real scaled = data[i] / scale;
         sum += scaled * scaled;
      }
      return scale * std::sqrt(sum);
   }
   else
   {
      GENDIL_VERIFY(scratch.Size() >= 1,
         "A nonempty device norm requires at least one reduction scratch entry.");
      Real* host = scratch.WriteHostData();
      Real* result = scratch.WriteDeviceData();
      if (scratch.Size() < 3)
      {
         KernelLoop<true>(size_t{1}, [=] GENDIL_HOST_DEVICE (size_t) mutable
         {
            vector_reduction_detail::ScaledSquares state;
            for (size_t i = 0; i < size; ++i) { state.Add(data[i]); }
            result[0] = state.Norm();
         });
      }
      else
      {
         const size_t parts = vector_reduction_detail::PartCount(
            size, (scratch.Size() - 1) / 2);
         Real* partial = result + 1;
         KernelLoop<true>(parts, [=] GENDIL_HOST_DEVICE (const size_t part) mutable
         {
            vector_reduction_detail::ScaledSquares state;
            for (size_t i = part; i < size; i += parts) { state.Add(data[i]); }
            partial[2 * part] = state.scale;
            partial[2 * part + 1] = state.sum;
         });
         KernelLoop<true>(size_t{1}, [=] GENDIL_HOST_DEVICE (size_t) mutable
         {
            vector_reduction_detail::ScaledSquares total;
            for (size_t p = 0; p < parts; ++p)
            {
               total.Merge({partial[2 * p], partial[2 * p + 1]});
            }
            result[0] = total.Norm();
         });
      }
      vector_reduction_detail::CopyResultsToHost(host, result, 1);
      return host[0];
   }
}

/** Write dot(basis(j), input) through the supplied callable, leaving input unchanged. */
template<class DotFunction, class Basis, class Input, class Coefficients>
GENDIL_HOST_DEVICE void ProjectFromDot(DotFunction&& dot, Basis&& basis,
   const Integer count, const Input& input, Coefficients& coefficients)
{
   for (Integer j = 0; j < count; ++j)
   {
      auto&& vector = basis(j);
      coefficients[j] = dot(vector, input);
   }
}

/**
 * @brief Project onto a Euclidean basis using batched reductions where scratch permits.
 *
 * Only an explicitly EuclideanDot is accepted. Contiguous storage changes the
 * scheduling, never the mathematical product. Coefficients are host-addressable;
 * device execution transfers only the active coefficient prefix. Scratch and
 * coefficient storage must not alias the basis or input.
 */
template<class KernelPolicy, class Basis, class Input, class Coefficients>
void ProjectEuclidean(const EuclideanDot<KernelPolicy>& dot, Basis&& basis,
   const Integer count, const Input& input, Coefficients& coefficients)
{
   if (count == 0) { return; }
   const size_t size = GetVectorSize(input);
   if (size == 0)
   {
      for (Integer j = 0; j < count; ++j) { coefficients[j] = 0.0; }
      return;
   }
   constexpr bool device = is_device_configuration_v<KernelPolicy>;
   constexpr bool contiguous = requires { basis.ContiguousStorage(); basis.ContiguousStride(); };
   const size_t columns = static_cast<size_t>(count);
   if constexpr (contiguous)
   {
      vector_reduction_detail::ValidateContiguousBasis(basis, columns, size);
   }
   if constexpr (!device)
   {
      if constexpr (contiguous)
      {
         auto storage = ReadKernelVector<false>(basis.ContiguousStorage());
         auto right = ReadKernelVector<false>(input);
         const size_t stride = basis.ContiguousStride();
         auto column_product = [&](const Integer j)
         {
            Real sum = 0.0;
            for (size_t i = 0; i < size; ++i)
            {
               sum += storage[static_cast<size_t>(j) * stride + i] * right[i];
            }
            return sum;
         };
         if constexpr (KernelWritableVector<false, Coefficients>)
         {
            // Acquire residency once, before workers write distinct entries.
            // The active subview preserves any coefficients beyond count.
            auto active = MakeVectorView(coefficients, size_t{0}, columns);
            auto destination = WriteKernelVector<false>(active);
            #pragma omp parallel for
            for (Integer j = 0; j < count; ++j) { destination[j] = column_product(j); }
         }
         else
         {
            // Generic scalar descriptors retain their strides and indexing.
            // Publishing in order also covers owner-affecting access proxies;
            // the independent column reductions still execute in parallel.
            #pragma omp parallel for ordered
            for (Integer j = 0; j < count; ++j)
            {
               const Real sum = column_product(j);
               #pragma omp ordered
               { coefficients[j] = sum; }
            }
         }
      }
      else { ProjectFromDot(dot, basis, count, input, coefficients); }
   }
   else
   {
      auto& scratch = dot.reduction_scratch;
      GENDIL_VERIFY(scratch.Size() >= 1,
         "A nonempty device projection requires at least one reduction scratch entry.");
      if (columns > scratch.Size() / 2)
      {
         ProjectFromDot(dot, basis, count, input, coefficients);
         return;
      }
      const size_t parts = vector_reduction_detail::PartCount(
         size, (scratch.Size() - columns) / columns);
      auto right = ReadKernelVector<true>(input);
      Real* host = scratch.WriteHostData();
      Real* result = scratch.WriteDeviceData();
      Real* partial = result + columns;
      if constexpr (contiguous)
      {
         auto storage = ReadKernelVector<true>(basis.ContiguousStorage());
         const size_t stride = basis.ContiguousStride();
         KernelLoop<true>(columns * parts,
            [=] GENDIL_HOST_DEVICE (const size_t task) mutable
         {
            const size_t column = task / parts;
            const size_t part = task % parts;
            Real sum = 0.0;
            for (size_t i = part; i < size; i += parts)
            {
               sum += storage[column * stride + i] * right[i];
            }
            partial[task] = sum;
         });
      }
      else
      {
         for (Integer j = 0; j < count; ++j)
         {
            auto&& vector = basis(j);
            GENDIL_VERIFY(GetVectorSize(vector) == size,
               "Euclidean projection requires matching vector extents.");
            auto left = ReadKernelVector<true>(vector);
            Real* column = partial + static_cast<size_t>(j) * parts;
            KernelLoop<true>(parts, [=] GENDIL_HOST_DEVICE (const size_t part) mutable
            {
               Real sum = 0.0;
               for (size_t i = part; i < size; i += parts) { sum += left[i] * right[i]; }
               column[part] = sum;
            });
         }
      }
      KernelLoop<true>(columns, [=] GENDIL_HOST_DEVICE (const size_t j) mutable
      {
         Real sum = 0.0;
         for (size_t p = 0; p < parts; ++p) { sum += partial[j * parts + p]; }
         result[j] = sum;
      });
      vector_reduction_detail::CopyResultsToHost(host, result, columns);
      for (Integer j = 0; j < count; ++j) { coefficients[j] = host[j]; }
   }
}

/** Add alpha times a contiguous basis combination in one host vector pass. */
template<bool OnDevice, class Basis, class Coefficients, class Output>
   requires (!OnDevice) && KernelWritableVector<false, Output> &&
      requires(Basis& basis)
      {
         ReadKernelVector<false>(basis.ContiguousStorage());
         basis.ContiguousStride();
      }
void HostDrivenFusedCombine(Basis&& basis, const Integer count,
   const Coefficients& coefficients, const Real alpha, Output& output)
{
   const size_t size = GetVectorSize(output);
   if (count == 0 || size == 0) { return; }
   vector_reduction_detail::ValidateContiguousBasis(basis, static_cast<size_t>(count), size);
   auto storage = ReadKernelVector<false>(basis.ContiguousStorage());
   auto destination = ReadWriteKernelVector<false>(output);
   const size_t stride = basis.ContiguousStride();
   #pragma omp parallel for
   for (size_t i = 0; i < size; ++i)
   {
      Real sum = 0.0;
      for (Integer j = 0; j < count; ++j)
      {
         sum += coefficients[j] * storage[static_cast<size_t>(j) * stride + i];
      }
      destination[i] += alpha * sum;
   }
}

/**
 * @brief Add a contiguous basis combination using one device kernel.
 *
 * Host-addressable coefficients are copied through caller-owned scratch of at
 * least count entries. Scratch must not alias the operands or coefficients;
 * basis and output must not overlap. Empty combinations need no scratch.
 */
template<bool OnDevice, class Basis, class Coefficients, class Output>
   requires OnDevice && KernelWritableVector<true, Output> &&
      requires(Basis& basis)
      {
         ReadKernelVector<true>(basis.ContiguousStorage());
         basis.ContiguousStride();
      }
void HostDrivenFusedCombine(Basis&& basis, const Integer count,
   const Coefficients& coefficients, const Real alpha, Output& output,
   Vector& coefficient_scratch)
{
   const size_t size = GetVectorSize(output);
   if (count == 0 || size == 0) { return; }
   const size_t columns = static_cast<size_t>(count);
   GENDIL_VERIFY(coefficient_scratch.Size() >= columns,
      "A fused device combination requires one scratch entry per coefficient.");
   vector_reduction_detail::ValidateContiguousBasis(basis, columns, size);
   auto storage = ReadKernelVector<true>(basis.ContiguousStorage());
   auto destination = ReadWriteKernelVector<true>(output);
   const size_t stride = basis.ContiguousStride();
   Real* host = coefficient_scratch.WriteHostData();
   for (Integer j = 0; j < count; ++j) { host[j] = coefficients[j]; }
   Real* device_coefficients = coefficient_scratch.WriteDeviceData();
   vector_reduction_detail::CopyCoefficientsToDevice(host, device_coefficients, columns);
   KernelLoop<true>(size, [=] GENDIL_HOST_DEVICE (const size_t i) mutable
   {
      Real sum = 0.0;
      for (size_t j = 0; j < columns; ++j)
      {
         sum += device_coefficients[j] * storage[j * stride + i];
      }
      destination[i] += alpha * sum;
   });
}

/**
 * @brief Publish a projection through a completed native-context dot callback.
 *
 * Every participant calls dot in the same order; only the primary participant
 * writes shared coefficients. The final barrier publishes them before return.
 */
template<class Context, class DotFunction, class Basis, class Input, class Coefficients>
GENDIL_HOST_DEVICE void LocalProject(Context& context, DotFunction&& dot,
   Basis&& basis, const Integer count, const Input& input, Coefficients& coefficients)
{
   for (Integer j = 0; j < count; ++j)
   {
      auto&& vector = basis(j);
      const Real value = dot(vector, input);
      if (IsPrimaryThread(context)) { coefficients[j] = value; }
   }
   if constexpr (is_threaded_v<Context>) { context.Synchronize(); }
}

/**
 * @brief Compute a robust Euclidean norm in a native execution context.
 *
 * Operands follow VisitLocal's distribution contract. A threaded device context
 * needs 2*Context::GetNumberOfThreads() temporary Real entries in its shared
 * arena. Every participant enters; the arena mark is restored before return.
 */
template<class Context, class Input>
GENDIL_HOST_DEVICE Real LocalScaledEuclideanNorm(Context& context, const Input& input)
{
   vector_reduction_detail::ScaledSquares local;
   VisitLocal(context, input, [&](auto... i) { local.Add(LocalEntry(input, i...)); });
#ifdef GENDIL_DEVICE_CODE
   if constexpr (is_threaded_v<Context>)
   {
      constexpr size_t threads = Context::GetNumberOfThreads();
      GENDIL_CHECK_MEMORY_ARENA_REQUEST(context.SharedAllocator, 2 * threads);
      const size_t mark = context.SharedAllocator.mark();
      Real* shared = context.SharedAllocator.allocate(2 * threads);
      GENDIL_ASSERT(shared != nullptr, "Local norm arena is exhausted.");
      const size_t thread = static_cast<size_t>(context.GetLinearThreadIndex());
      shared[2 * thread] = local.scale;
      shared[2 * thread + 1] = local.sum;
      context.Synchronize();
      vector_reduction_detail::ScaledSquares total;
      for (size_t t = 0; t < threads; ++t)
      {
         total.Merge({shared[2 * t], shared[2 * t + 1]});
      }
      context.Synchronize();
      context.SharedAllocator.restore(mark);
      return total.Norm();
   }
#endif
   return local.Norm();
}

/**
 * @brief Batch Euclidean projections in a native execution context.
 *
 * This is an explicit Euclidean operation, not a customization of an arbitrary
 * supplied dot. A threaded device context needs count temporary Real entries;
 * basis, input, and coefficients have matching collective ownership. Every
 * participant enters, coefficients are published, and arena storage is restored.
 */
template<class Context, class Basis, class Input, class Coefficients>
GENDIL_HOST_DEVICE void LocalProjectEuclidean(Context& context, Basis&& basis,
   const Integer count, const Input& input, Coefficients& coefficients)
{
   GENDIL_ASSERT(count >= 0, "Projection requires a nonnegative basis count.");
   if (count <= 0) { return; }
#ifdef GENDIL_DEVICE_CODE
   if constexpr (is_threaded_v<Context>)
   {
      const size_t mark = context.SharedAllocator.mark();
      Real* shared = context.SharedAllocator.allocate(static_cast<size_t>(count));
      GENDIL_ASSERT(shared != nullptr, "Local projection arena is exhausted.");
      if (IsPrimaryThread(context))
      {
         for (Integer j = 0; j < count; ++j) { shared[j] = 0.0; }
      }
      context.Synchronize();
      for (Integer j = 0; j < count; ++j)
      {
         auto&& vector = basis(j);
         Real local = 0.0;
         VisitLocal(context, input, [&](auto... i)
         {
            local += LocalEntry(vector, i...) * LocalEntry(input, i...);
         });
         AtomicAddInPlace(shared[j], local);
      }
      context.Synchronize();
      if (IsPrimaryThread(context))
      {
         for (Integer j = 0; j < count; ++j) { coefficients[j] = shared[j]; }
      }
      context.Synchronize();
      context.SharedAllocator.restore(mark);
      return;
   }
#endif
   for (Integer j = 0; j < count; ++j)
   {
      auto&& vector = basis(j);
      coefficients[j] = LocalDot(context, vector, input);
   }
}

} // namespace gendil
