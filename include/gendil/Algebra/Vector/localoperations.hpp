// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/** @file @brief Vector arithmetic inside a native kernel context. */

#include "gendil/Algebra/Vector/operations.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/Utilities/KernelContext/threadedshapecoverage.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/reverseindexsequence.hpp"
#include "gendil/Utilities/Loop/threadloop.hpp"
#include "gendil/Utilities/Loop/unitloop.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"

#include <concepts>
#include <iterator>
#include <type_traits>

namespace gendil
{

namespace details
{

template<class VectorType>
inline constexpr bool is_threaded_operand_v = requires
{
   typename std::remove_cvref_t<VectorType>::register_shape;
   typename std::remove_cvref_t<VectorType>::thread_shape;
};

template<class VectorType>
constexpr bool WritableLocalEntries()
{
   if constexpr (is_threaded_operand_v<VectorType>)
   {
      return []<size_t... I>(std::index_sequence<I...>)
      {
         return requires(VectorType& vector)
         {
            vector.data((void(I), size_t{})...) = Real{};
         };
      }(std::make_index_sequence<
           std::remove_cvref_t<VectorType>::register_shape::size()>{});
   }
   else
   {
      return requires(VectorType& vector) { vector[0] = Real{}; } ||
             requires(VectorType& vector) { vector(0) = Real{}; };
   }
}

template<class VectorType>
constexpr bool ReadableLocalEntries()
{
   if constexpr (is_threaded_operand_v<VectorType>)
   {
      return []<size_t... I>(std::index_sequence<I...>)
      {
         return requires(const VectorType& vector)
         {
            { vector.data((void(I), size_t{})...) } -> std::convertible_to<Real>;
         };
      }(std::make_index_sequence<
           std::remove_cvref_t<VectorType>::register_shape::size()>{});
   }
   else
   {
      return requires(const VectorType& vector)
             {
                { vector[0] } -> std::convertible_to<Real>;
             } || requires(const VectorType& vector)
             {
                { vector(0) } -> std::convertible_to<Real>;
             };
   }
}

} // namespace details

/// Identify the unique participant that publishes shared scalar storage.
template<class Context>
GENDIL_HOST_DEVICE bool IsPrimaryThread(Context& context)
{
   if constexpr (is_threaded_v<Context>)
   {
      return context.GetLinearThreadIndex() == 0;
   }
   else
   {
      return true;
   }
}

/// Access a local register entry or a rank-one array/view entry without copying.
template<class VectorType, class... Indices>
GENDIL_HOST_DEVICE decltype(auto) LocalEntry(VectorType& vector, Indices... indices)
{
   if constexpr (details::is_threaded_operand_v<VectorType>)
   {
      return vector.data(indices...);
   }
   else if constexpr (requires { vector(indices...); })
   {
      return vector(indices...);
   }
   else
   {
      static_assert(sizeof...(indices) == 1);
      return vector[(indices, ...)];
   }
}

/// Return the extent of a rank-one local array or generic vector view.
template<class VectorType>
GENDIL_HOST_DEVICE size_t LocalSize(const VectorType& vector)
{
   if constexpr (requires { GetVectorSize(vector); })
   {
      return GetVectorSize(vector);
   }
   else if constexpr (requires { std::size(vector); })
   {
      return std::size(vector);
   }
   else
   {
      static_assert(std::remove_cvref_t<VectorType>::rank == 1);
      return std::extent_v<decltype(vector.data)>;
   }
}

/**
 * Visit entries owned by the current participant, without synchronization.
 *
 * Rank-one shared arrays/views partition by the native linear thread index.
 * ThreadedView operands visit their register shape and require a native thread
 * layout covering the declared thread shape. All operands of a local operation
 * must have the same logical shape and distribution and be directly accessible
 * in the current memory space. No ownership or residency adaptation is performed.
 */
template<class Context, class VectorType, class Function>
GENDIL_HOST_DEVICE void VisitLocal(
   Context& context, const VectorType& vector, Function&& function)
{
   if constexpr (details::is_threaded_operand_v<VectorType>)
   {
      using Vector = std::remove_cvref_t<VectorType>;
#ifdef GENDIL_DEVICE_CODE
      if constexpr (is_threaded_v<Context>)
      {
         static_assert(threaded_shape_covered_v<Context,
            typename Vector::thread_shape>,
            "The native thread layout must cover the operand thread shape.");
         ThreadLoop<typename Vector::thread_shape>(context, [&](auto...)
         {
            UnitLoop<typename Vector::register_shape>(function);
         });
      }
      else
#endif
      {
         UnitLoop<typename Vector::register_shape>(function);
      }
   }
   else
   {
      size_t first = 0;
      size_t step = 1;
      if constexpr (is_threaded_v<Context>)
      {
         first = context.GetLinearThreadIndex();
         step = Context::GetNumberOfThreads();
      }
      for (size_t i = first; i < LocalSize(vector); i += step)
      {
         function(i);
      }
   }
}

/// Complete y += alpha*x; threaded participants must call uniformly.
template<class Context, class Input, class Output>
   requires(details::ReadableLocalEntries<Input>() &&
            details::WritableLocalEntries<Output>())
GENDIL_HOST_DEVICE void Axpy(
   Context& context, const Real alpha, const Input& input, Output& output)
{
   VisitLocal(context, output, [&](auto... i)
   {
      LocalEntry(output, i...) += alpha * LocalEntry(input, i...);
   });
   if constexpr (is_threaded_v<Context>) { context.Synchronize(); }
}

/// Complete a copy into local storage; threaded participants call uniformly.
template<class Context, class Input, class Output>
   requires(details::ReadableLocalEntries<Input>() &&
            details::WritableLocalEntries<Output>())
GENDIL_HOST_DEVICE void LocalCopy(Context& context, const Input& input, Output& output)
{
   VisitLocal(context, output, [&](auto... i)
   {
      LocalEntry(output, i...) = LocalEntry(input, i...);
   });
   if constexpr (is_threaded_v<Context>) { context.Synchronize(); }
}

/// Complete zeroing local storage; threaded participants call uniformly.
template<class Context, class Output>
   requires(details::WritableLocalEntries<Output>())
GENDIL_HOST_DEVICE void LocalZero(Context& context, Output& output)
{
   VisitLocal(context, output, [&](auto... i)
   {
      LocalEntry(output, i...) = Real{};
   });
   if constexpr (is_threaded_v<Context>) { context.Synchronize(); }
}

/// Divide directly by length; no norm reduction or reciprocal is substituted.
template<class Context, class Output>
   requires(details::WritableLocalEntries<Output>())
GENDIL_HOST_DEVICE void LocalDivide(
   Context& context, Output& output, const Real length)
{
   VisitLocal(context, output, [&](auto... i)
   {
      LocalEntry(output, i...) /= length;
   });
   if constexpr (is_threaded_v<Context>) { context.Synchronize(); }
}

/// Complete a fused local basis update using uniformly readable coefficients.
template<class Context, class Basis, class Coefficients, class Output>
   requires(details::WritableLocalEntries<Output>())
GENDIL_HOST_DEVICE void LocalCombine(Context& context, Basis& basis,
   const Integer count, const Coefficients& coefficients, const Real alpha,
   Output& output)
{
   VisitLocal(context, output, [&](auto... i)
   {
      Real sum = 0;
      for (Integer j = 0; j < count; ++j)
      {
         auto&& vector = basis(j);
         sum += coefficients[j] * LocalEntry(vector, i...);
      }
      LocalEntry(output, i...) += alpha * sum;
   });
   if constexpr (is_threaded_v<Context>) { context.Synchronize(); }
}

/**
 * Return the Euclidean product, completing its own operand/scratch dependencies.
 *
 * CPU execution is thread-local, not a host-thread collective. A threaded device
 * call is collective within one native work item, requires one available Real in
 * SharedAllocator, and returns the same scalar to every participant. Unthreaded
 * execution uses private accumulation and introduces no barriers.
 */
template<class Context, class Input, class Other>
   requires(details::ReadableLocalEntries<Input>() &&
            details::ReadableLocalEntries<Other>())
GENDIL_HOST_DEVICE Real LocalDot(
   Context& context, const Input& input, const Other& other)
{
   Real value = 0;
   VisitLocal(context, input, [&](auto... i)
   {
      value += LocalEntry(input, i...) * LocalEntry(other, i...);
   });
#ifdef GENDIL_DEVICE_CODE
   if constexpr (is_threaded_v<Context>)
   {
      const auto mark = context.SharedAllocator.mark();
      Real* result = context.SharedAllocator.allocate(1);
      GENDIL_ASSERT(result != nullptr, "LocalDot requires one shared Real slot.");
      if (IsPrimaryThread(context)) { *result = 0; }
      context.Synchronize();
      AtomicAddInPlace(*result, value);
      context.Synchronize();
      value = *result;
      context.Synchronize();
      context.SharedAllocator.restore(mark);
   }
#endif
   return value;
}

} // namespace gendil
