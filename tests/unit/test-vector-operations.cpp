// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Algebra/Vector/localoperations.hpp>
#include <gendil/Utilities/KernelContext/KernelConfigurations/host.hpp>
#include <gendil/Utilities/MathHelperFunctions/isfinite.hpp>
#include <gendil/Utilities/RecursiveArray/recursivearray.hpp>
#include <gendil/FiniteElementMethod/ShapeFunctions/finiteelementorders.hpp>
#include <gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/elementdof.hpp>
#include <gendil/Utilities/View/threadedview.hpp>
#include <gendil/Utilities/View/Layouts/stridedlayout.hpp>

#include <array>
#include <atomic>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>

#if defined(GENDIL_USE_OPENMP) && defined(_OPENMP)
#include <omp.h>
#endif

using namespace gendil;

namespace
{

static_assert(
   std::is_same_v<
      decltype(Zero<false>(std::declval<Vector&>())),
      void>);

bool Check(const bool condition, const char* message)
{
   if (!condition)
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool TestGenDiLVectorZero()
{
   Vector vector(3);
   auto* host_data = WriteHostVector(vector);
   host_data[0] = Real(2);
   host_data[1] = Real(4);
   host_data[2] = Real(6);

   Zero<false>(vector);
   bool success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "Zero<false> produced the wrong GenDiL validity state.");
   const auto* host_zero = ReadHostVector(vector);
   success = Check(
      host_zero[0] == Real(0) &&
      host_zero[1] == Real(0) &&
      host_zero[2] == Real(0),
      "Zero<false> did not zero the GenDiL vector.") && success;

   host_data = WriteHostVector(vector);
   host_data[0] = Real(3);
   host_data[1] = Real(5);
   host_data[2] = Real(7);
   Zero<true>(vector);
#ifdef GENDIL_USE_DEVICE
   success = Check(
      !vector.IsHostValid() && vector.IsDeviceValid(),
      "Zero<true> produced the wrong GenDiL validity state.") && success;
#else
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU Zero<true> should use GenDiL host storage.") && success;
#endif
   const auto* device_zero = ReadHostVector(vector);
   success = Check(
      device_zero[0] == Real(0) &&
      device_zero[1] == Real(0) &&
      device_zero[2] == Real(0),
      "Zero<true> did not zero the GenDiL vector.") && success;

   Vector empty;
   Zero<false>(empty);
   Zero<true>(empty);
   success = Check(
      GetVectorSize(empty) == size_t(0),
      "Zero changed an empty GenDiL vector.") && success;

   return success;
}

bool TestVectorSubviewZero()
{
   Vector vector(5);
   vector = Real(7);
   auto middle = MakeVectorView(vector, size_t(1), size_t(3));
   Zero<true>(middle);

   const auto* values = ReadHostVector(vector);
   bool success = Check(
      values[0] == Real(7) &&
      values[1] == Real(0) &&
      values[2] == Real(0) &&
      values[3] == Real(0) &&
      values[4] == Real(7),
      "Device-selected zero changed values outside a vector subview.");

   auto device_values = ReadWriteKernelVector<true>(vector);
   KernelLoop<true>(size_t{5}, [=] GENDIL_HOST_DEVICE (size_t i) mutable
   {
      device_values[i] = Real(i + 10);
   });
   auto nested = MakeVectorView(middle, size_t{1}, size_t{1});
   Zero<false>(nested);
   values = ReadHostVector(vector);
   return Check(values[0] == 10 && values[1] == 11 && values[2] == 0 &&
      values[3] == 13 && values[4] == 14,
      "Host-selected zero lost current device entries outside its subview.") && success;
}

template<class Context, class Input, class Output>
concept HasNativeAxpy = requires(Context& context, const Input& x, Output& y)
{
   Axpy(context, Real{2}, x, y);
};

struct PrivateContext : HostKernelConfiguration
{
   void Synchronize() = delete;
};

using LocalArray = std::array<Real, 4>;
using ReadOnlyView = decltype(MakeVectorView(static_cast<const Real*>(nullptr), 4));
static_assert(HasNativeAxpy<PrivateContext, LocalArray, LocalArray>);
static_assert(!HasNativeAxpy<PrivateContext, LocalArray, const LocalArray>);
static_assert(!HasNativeAxpy<PrivateContext, LocalArray, ReadOnlyView>);

struct NoncopyableStorage
{
   LocalArray values{1, 2, 3, 4};
   NoncopyableStorage() = default;
   NoncopyableStorage(const NoncopyableStorage&) = delete;
   size_t size() const { return values.size(); }
   Real& operator[](size_t i) { return values[i]; }
   const Real& operator[](size_t i) const { return values[i]; }
};

bool TestLocalArrayOperations()
{
   HostKernelConfiguration context;
   const std::array<Real, 3> input{1, 2, 3};
   std::array<Real, 3> output{};
   LocalCopy(context, input, output);
   Axpy(context, Real(2), input, output);
   LocalDivide(context, output, Real(3));
   bool success = Check(LocalDot(context, input, output) == 14,
      "Host-context array copy/Axpy/divide/dot changed its result.");
   LocalZero(context, output);
   std::array<std::array<Real, 3>, 2> vectors{{{1, 2, 3}, {3, 2, 1}}};
   auto basis = MakeVectorView(vectors.data(), vectors.size());
   const std::array<Real, 2> coefficients{2, -1};
   LocalCombine(context, basis, 2, coefficients, Real(1), output);
   return Check(output == std::array<Real, 3>{-1, 2, 5} &&
      LocalDot(context, input, output) == 18,
      "Local zero/combine changed the array-basis result.") && success;
}

bool TestAxpyFamiliesAndLocalOperations()
{
   Vector source(6), destination(6);
   WriteHostVector(source);
   WriteHostVector(destination);
   for (size_t i = 0; i < 6; ++i)
   {
      source[i] = Real(i + 1);
      destination[i] = 1;
   }
   Axpy(Real{2}, source, destination);
   bool success = Check(destination[5] == 13,
      "The original owning Axpy changed its result.");
   auto x = MakeVectorView(source, 1, 3);
   auto y = MakeVectorView(destination, 2, 3);
   Axpy<false>(2, x, y);
   Divide<false>(y, 2);
   success = Check(destination[1] == 5 && destination[2] == 5.5 &&
      destination[4] == 9.5 && destination[5] == 13,
      "Explicit Axpy/division changed owner subview addressing.") && success;
   Axpy<true>(1, x, y);
   success = Check(destination[2] == 7.5 && destination[4] == 13.5,
      "Explicit device-selected Axpy changed its result.") && success;

   PrivateContext context;
   NoncopyableStorage storage;
   auto borrowed = MakeView(ReferenceContainer{storage}, VectorLayout{4});
   const LocalArray ones{1, 1, 1, 1};
   Axpy(context, 3, ones, borrowed);
   LocalDivide(context, borrowed, 2);
   success = Check(storage[3] == 3.5,
      "Native Axpy/division did not preserve borrowed ownership.") && success;
   LocalCopy(context, ones, borrowed);
   success = Check(LocalDot(context, ones, borrowed) == 4,
      "Local copy/dot returned the wrong value.") && success;
   LocalZero(context, borrowed);
   success = Check(storage[0] == 0 && storage[3] == 0,
      "Local zero did not clear borrowed storage.") && success;

   const Real tiny = std::numeric_limits<Real>::denorm_min() * 16;
   LocalArray small{tiny, tiny, tiny, tiny};
   LocalDivide(context, small, tiny);
   Vector small_owner(1);
   WriteHostVector(small_owner)[0] = tiny;
   Divide<false>(small_owner, tiny);
   success = Check(small == ones && small_owner[0] == 1,
      "Division must not substitute an overflowing reciprocal multiplication.") && success;

   SerialRecursiveArray<Real, 4> recursive_x{}, recursive_y{};
   recursive_x = 2;
   recursive_y = 1;
   Axpy(Real{3}, recursive_x, recursive_y);
   Axpy(context, 2, recursive_x, recursive_y);
   success = Check(recursive_y(2) == 11,
      "Recursive-array Axpy overloads changed behavior.") && success;

   using Registers = ThreadedView<std::index_sequence<2, 3>,
      HostKernelConfiguration, SerialRecursiveArray<Real, 2, 3>>;
   Registers register_x{}, register_y{};
   register_x = 2;
   register_y = 1;
   Axpy(Real{3}, register_x, register_y);
   Axpy(context, 2, register_x, register_y);
   success = Check(register_y.data(1, 2) == 11 &&
      LocalDot(context, register_x, register_y) == 132,
      "ThreadedView Axpy/dot lost register-shape coverage.") && success;
   static_assert(!HasNativeAxpy<PrivateContext, Registers, const Registers>);

   std::array<LocalArray, 2> basis{{LocalArray{1, 2, 3, 4}, LocalArray{5, 6, 7, 8}}};
   auto references = [&](Integer j) -> LocalArray& { return basis[size_t(j)]; };
   const std::array<Real, 2> coefficients{2, -1};
   LocalArray fused{}, descriptor_result{};
   LocalCombine(context, references, 2, coefficients, 1, fused);
   auto local_descriptors = [&](Integer j)
   {
      return MakeVectorView(basis[size_t(j)].data(), 4);
   };
   LocalCombine(context, local_descriptors, 2, coefficients, 1, descriptor_result);
   Vector basis_owner(8), reference(4);
   WriteHostVector(basis_owner);
   for (size_t i = 0; i < 8; ++i) { basis_owner[i] = Real(i + 1); }
   auto descriptors = [&](Integer j) { return MakeVectorView(basis_owner, 4 * j, 4); };
   Zero<false>(reference);
   gendil::Combine<false>(descriptors, 2, coefficients, 1, reference);
   success = Check(fused == LocalArray{-3, -2, -1, 0} && descriptor_result == fused &&
      reference[0] == fused[0] && reference[3] == fused[3],
      "Reference/fused combination did not retain basis references/descriptors.") && success;

   const Real maximum = std::numeric_limits<Real>::max();
   return Check(IsPrimaryThread(context) && IsFinite(maximum) && IsFinite(-maximum) &&
      !IsFinite(std::numeric_limits<Real>::infinity()) &&
      !IsFinite(std::numeric_limits<Real>::quiet_NaN()),
      "Shared scalar finite predicate failed.") && success;
}

#if defined(GENDIL_USE_OPENMP) && defined(_OPENMP)
// This models partitioned mutation/completion only, not a CPU collective dot.
struct SharedParticipant
{
   static constexpr bool is_host_configuration = false;
   static constexpr size_t thread_block_dim = 1;
   static constexpr size_t GetNumberOfThreads() { return 4; }
   size_t lane;
   size_t GetLinearThreadIndex() const { return lane; }
   void Synchronize()
   {
      #pragma omp barrier
   }
};

bool TestStridedNativeCompletion()
{
   std::array<Real, 32> source, destination;
   for (size_t i = 0; i < source.size(); ++i)
   {
      source[i] = Real(i + 1);
      destination[i] = -7;
   }
   auto x = MakeView(source.data(), MakeOffsetLayout(
      StridedLayout<1, DynamicExtents<1>>{std::array{3}, DynamicExtents<1>{7}}, 2));
   auto y = MakeView(destination.data(), MakeOffsetLayout(
      StridedLayout<1, DynamicExtents<1>>{std::array{2}, DynamicExtents<1>{7}}, 5));
   auto active_x = MakeView(ReferenceContainer{x}, VectorLayout{5});
   auto active_y = MakeView(ReferenceContainer{y}, VectorLayout{5});
   const int dynamic = omp_get_dynamic();
   omp_set_dynamic(0);
   int team_size = 0;
   std::atomic<bool> completed{true};
   #pragma omp parallel num_threads(4)
   {
      const auto lane = static_cast<size_t>(omp_get_thread_num());
      if (lane == 0) { team_size = omp_get_num_threads(); }
      if (omp_get_num_threads() == 4)
      {
         SharedParticipant context{lane};
         Axpy(context, 2, active_x, active_y);
         // Observe completion before the parallel region's implicit barrier.
         for (size_t j = 0; j < 5; ++j)
         {
            if (active_y[j] != -7 + 2 * x[j]) { completed = false; }
         }
      }
   }
   omp_set_dynamic(dynamic);
   if (!Check(team_size == 4,
      "Native completion probe requires four OpenMP participants."))
   {
      return false;
   }
   bool success = Check(completed,
      "Native Axpy returned before all shared-vector entries were published.");
   for (size_t i = 0; i < destination.size(); ++i)
   {
      Real expected = -7;
      for (size_t j = 0; j < 5; ++j)
      {
         if (i == 5 + 2 * j) { expected += 2 * source[2 + 3 * j]; }
      }
      success = Check(destination[i] == expected,
         "Native Axpy changed inactive or strided-prefix storage.") && success;
   }
   return success;
}
#endif

#ifdef GENDIL_USE_MFEM
bool TestMFEMVectorZero()
{
   mfem::Vector vector(3);
   auto* data = WriteHostVector(vector);
   data[0] = 4.0;
   data[1] = 5.0;
   data[2] = 6.0;

   Zero<false>(vector);
   const auto* read = ReadHostVector(vector);
   return Check(
      read[0] == 0.0 && read[1] == 0.0 && read[2] == 0.0,
      "Zero<false> did not zero the MFEM vector.");
}
#endif

} // namespace

int main()
{
   bool success = TestGenDiLVectorZero();
   success = TestVectorSubviewZero() && success;
   success = TestLocalArrayOperations() && success;
   success = TestAxpyFamiliesAndLocalOperations() && success;
#if defined(GENDIL_USE_OPENMP) && defined(_OPENMP)
   success = TestStridedNativeCompletion() && success;
#else
   std::cout << "Skipping native completion probe: OpenMP is not enabled for this source.\n";
#endif
#ifdef GENDIL_USE_MFEM
   success = TestMFEMVectorZero() && success;
#endif
   return success ? 0 : 1;
}
