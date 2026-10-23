// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Algebra/Vector/reductions.hpp>
#include <gendil/Algebra/Vector/slices.hpp>
#include <gendil/Utilities/View/Layouts/stridedlayout.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>

using namespace gendil;

namespace {

using DevicePolicy = DeviceKernelConfiguration<ThreadBlockLayout<>, 0>;

bool Check(const bool success, const char* message)
{
   if (!success) { std::cerr << message << '\n'; }
   return success;
}

bool Close(const Real actual, const Real expected, const Real tolerance = 1e-12)
{
   return std::abs(actual - expected) <= tolerance * std::max(Real{1}, std::abs(expected));
}

bool TestCheckedSqrt()
{
   bool success = Check(CheckedSqrt(0) == 0 && CheckedSqrt(25) == 5,
      "CheckedSqrt changed the square root of a valid input.");
   for (const Real invalid : {Real{-1}, std::numeric_limits<Real>::quiet_NaN(),
         std::numeric_limits<Real>::infinity(), -std::numeric_limits<Real>::infinity()})
   {
      success = Check(std::isnan(CheckedSqrt(invalid)),
         "CheckedSqrt accepted a negative or nonfinite input.") && success;
   }
   const std::array<Real, 1> input{};
   auto zero_dot = [](const auto&, const auto&) { return Real{0}; };
   auto nan_dot = [](const auto&, const auto&) { return std::numeric_limits<Real>::quiet_NaN(); };
   success = Check(NormFromDot(zero_dot, input) == 0 && std::isnan(NormFromDot(nan_dot, input)),
      "NormFromDot changed its zero or NaN self-product behavior.") && success;
   return success;
}

template<class Policy>
bool TestDotAndNorm(const size_t scratch_size)
{
   constexpr size_t size = 1031;
   Vector first(size), second(size), scratch(scratch_size);
   Real* left = first.WriteHostData();
   Real* right = second.WriteHostData();
   Real expected_dot = 0.0;
   Real expected_square = 0.0;
   for (size_t i = 0; i < size; ++i)
   {
      left[i] = Real(i % 11) - 4.0;
      right[i] = Real(i % 7) + 0.25;
      expected_dot += left[i] * right[i];
      expected_square += left[i] * left[i];
   }
   if constexpr (is_device_configuration_v<Policy>)
   {
      (void)first.ReadWriteDeviceData();
      (void)second.ReadWriteDeviceData();
   }
   EuclideanDot<Policy> dot{scratch};
   bool success = Check(Close(dot(first, second), expected_dot), "Scratch-backed dot is incorrect.");
   Vector small(2);
   auto* small_values = small.WriteHostData();
   small_values[0] = 3;
   small_values[1] = 4;
   success = Check(dot(small, small) == 25 && ScaledEuclideanNorm(dot, small) == 5 &&
      Dot<Policy>(scratch, small, small) == 25,
      "Small exact dot/norm reduction changed its result.") && success;
   success = Check(Close(Dot<Policy>(scratch, first, second), expected_dot),
      "Standalone scratch-backed Dot is incorrect.") && success;
   success = Check(Close(ScaledEuclideanNorm(dot, first), std::sqrt(expected_square)),
      "Scaled Euclidean norm is incorrect.") && success;
#ifdef GENDIL_USE_DEVICE
   if constexpr (is_device_configuration_v<Policy>)
   {
      success = Check(!first.IsHostValid() && first.IsDeviceValid() &&
         !second.IsHostValid() && second.IsDeviceValid(),
         "A device reduction transferred a full operand to the host.") && success;
   }
#endif
   Vector empty;
   Vector no_scratch;
   EuclideanDot<Policy> empty_dot{no_scratch};
   success = Check(empty_dot(empty, empty) == 0.0 &&
      ScaledEuclideanNorm(empty_dot, empty) == 0.0,
      "Empty reductions should need no scratch.") && success;

   Vector extreme(2);
   auto* values = extreme.WriteHostData();
   values[0] = 3e200;
   values[1] = 4e200;
   success = Check(Close(ScaledEuclideanNorm(dot, extreme) / 1e200, 5.0),
      "Scaled Euclidean norm overflowed a representable result.") && success;
   success = Check(std::isnan(NormFromDot(dot, extreme)),
      "NormFromDot should reject an overflowing self-dot.") && success;
   values = extreme.WriteHostData();
   values[0] = 3e-200;
   values[1] = 4e-200;
   success = Check(Close(ScaledEuclideanNorm(dot, extreme) / 1e-200, 5.0),
      "Scaled Euclidean norm underflowed a representable result.") && success;
   values = extreme.WriteHostData();
   values[0] = std::numeric_limits<Real>::infinity();
   values[1] = 0.0;
   success = Check(std::isnan(ScaledEuclideanNorm(dot, extreme)),
      "Scaled Euclidean norm accepted infinity.") && success;
   values = extreme.WriteHostData();
   values[0] = std::numeric_limits<Real>::quiet_NaN();
   values[1] = 1.0;
   success = Check(std::isnan(ScaledEuclideanNorm(dot, extreme)),
      "Scaled Euclidean norm accepted NaN.") && success;
   return success;
}

template<class Policy>
bool TestProjection(const size_t scratch_size)
{
   constexpr size_t size = 1031;
   constexpr size_t count = 3;
   Vector storage(size * count), input(size), scratch(scratch_size);
   auto basis = MakeVectorSlices(storage, size, count);
   auto* vectors = storage.WriteHostData();
   auto* values = input.WriteHostData();
   std::array<Real, count> expected{};
   for (size_t i = 0; i < size; ++i)
   {
      values[i] = Real(i % 7) - 2.0;
      for (size_t j = 0; j < count; ++j)
      {
         const Real entry = Real((i + j) % 11) - 3.0;
         vectors[j * size + i] = entry;
         expected[j] += entry * values[i];
      }
   }
   EuclideanDot<Policy> dot{scratch};
   std::array<Real, count + 1> coefficients{0.0, 0.0, 0.0, -99.0};
   ProjectEuclidean(dot, basis, count, input, coefficients);
   bool success = true;
   for (size_t j = 0; j < count; ++j)
   {
      success = Check(Close(coefficients[j], expected[j]),
         "Contiguous Euclidean projection is incorrect.") && success;
   }
   success = Check(coefficients[count] == -99.0,
      "Projection overwrote inactive coefficients.") && success;

   // The same operation also supports noncontiguous access returning shallow
   // descriptors by value. It never substitutes an owning-vector copy.
   auto generic_basis = [&](const Integer j) { return basis(j); };
   coefficients = {0.0, 0.0, 0.0, -99.0};
   ProjectEuclidean(dot, generic_basis, count, input, coefficients);
   for (size_t j = 0; j < count; ++j)
   {
      success = Check(Close(coefficients[j], expected[j]),
         "Generic Euclidean projection is incorrect.") && success;
   }
   coefficients[0] = -17.0;
   ProjectEuclidean(dot, generic_basis, 0, input, coefficients);
   success = Check(coefficients[0] == -17.0,
      "An empty projection changed coefficients.") && success;
   Vector empty;
   Vector empty_scratch;
   EuclideanDot<Policy> empty_dot{empty_scratch};
   ProjectEuclidean(empty_dot, generic_basis, count, empty, coefficients);
   success = Check(coefficients[0] == 0.0 && coefficients[1] == 0.0 && coefficients[2] == 0.0,
      "Projection of an empty input should produce zero coefficients.") && success;
   return success;
}

template<bool OnDevice>
bool TestFusedCombination()
{
   constexpr size_t size = 1031;
   constexpr size_t count = 3;
   Vector storage(size * count), output(size), scratch(RequiredCombinationWorkspaceSize(count));
   auto basis = MakeVectorSlices(storage, size, count);
   auto* vectors = storage.WriteHostData();
   auto* values = output.WriteHostData();
   const std::array<Real, count> coefficients{2.0, -3.0, 0.5};
   std::array<Real, size> expected{};
   for (size_t i = 0; i < size; ++i)
   {
      values[i] = Real(i % 5);
      expected[i] = values[i];
      for (size_t j = 0; j < count; ++j)
      {
         vectors[j * size + i] = Real((i + j) % 11) - 3.0;
         expected[i] += -0.25 * coefficients[j] * vectors[j * size + i];
      }
   }
   if constexpr (OnDevice)
   {
      HostDrivenFusedCombine<true>(basis, count, coefficients, -0.25, output, scratch);
   }
   else { HostDrivenFusedCombine<false>(basis, count, coefficients, -0.25, output); }
   const auto* actual = output.ReadHostData();
   bool success = true;
   for (size_t i = 0; i < size; ++i)
   {
      success = Check(Close(actual[i], expected[i]), "Fused combination is incorrect.") && success;
   }
   return success;
}

bool TestProjectionCoefficientOwners()
{
   constexpr size_t size = 17;
   constexpr size_t count = 64;
   constexpr Real sentinel = -99.0;
   Vector storage(size * count), input(size), scratch;
   auto basis = MakeVectorSlices(storage, size, count);
   auto* vectors = storage.WriteHostData();
   auto* values = input.WriteHostData();
   std::array<Real, count> expected{};
   for (size_t i = 0; i < size; ++i)
   {
      values[i] = Real(i + 1);
      for (size_t j = 0; j < count; ++j)
      {
         vectors[j * size + i] = Real((i + j) % 7) - 2.0;
         expected[j] += vectors[j * size + i] * values[i];
      }
   }
   EuclideanDot<HostKernelConfiguration> dot{scratch};
   Vector owning(count + 2);
   owning = sentinel;
   ProjectEuclidean(dot, basis, count, input, owning);
   const auto* actual = owning.ReadHostData();
   bool success = true;
   for (size_t j = 0; j < count; ++j)
   {
      success = Check(actual[j] == expected[j],
         "Projection into owning coefficients is incorrect.") && success;
   }
   success = Check(actual[count] == sentinel && actual[count + 1] == sentinel,
      "Projection into an owner overwrote its inactive coefficient suffix.") && success;

   owning = sentinel;
   auto middle = MakeVectorView(owning, size_t{1}, count);
   ProjectEuclidean(dot, basis, count, input, middle);
   actual = owning.ReadHostData();
   for (size_t j = 0; j < count; ++j)
   {
      success = Check(actual[j + 1] == expected[j],
         "Projection into owner-bound coefficient subview is incorrect.") && success;
   }
   success = Check(actual[0] == sentinel && actual[count + 1] == sentinel,
      "Projection into a coefficient subview changed outside entries.") && success;

   // A generic scalar layout is not contiguous kernel-vector access. Preserve
   // its addressing and avoid concurrent owner-validity writes through its proxy.
   Vector strided_owner(2 * count);
   strided_owner = sentinel;
   auto strided = MakeView(ReferenceContainer{strided_owner},
      StridedLayout<1, DynamicExtents<1>>{
         std::array<GlobalIndex, 1>{2}, DynamicExtents<1>{count}});
   ProjectEuclidean(dot, basis, count, input, strided);
   actual = strided_owner.ReadHostData();
   for (size_t j = 0; j < count; ++j)
   {
      success = Check(actual[2 * j] == expected[j] && actual[2 * j + 1] == sentinel,
         "Projection lost scalar coefficient strides or modified gap entries.") && success;
   }
   return success;
}

using LocalVector = std::array<Real, 3>;
struct WeightedDot
{
   int calls = 0;
   WeightedDot() = default;
   WeightedDot(const WeightedDot&) = delete;
   Real operator()(const LocalVector& first, const LocalVector& second)
   {
      ++calls;
      return first[0] * second[0] + 2.0 * first[1] * second[1] + 3.0 * first[2] * second[2];
   }
};

bool TestSuppliedDotAndLocalOperations()
{
   Real arena[1]{};
   KernelContext<HostKernelConfiguration, 0> context(arena);
   LocalVector vectors[2]{{1, 0, 1}, {0, 1, 1}};
   const LocalVector input{2, 3, 4};
   auto basis = [&](Integer j) -> LocalVector& { return vectors[j]; };
   std::array<Real, 2> coefficients{};
   WeightedDot dot;
   ProjectFromDot(dot, basis, 2, input, coefficients);
   bool success = Check(dot.calls == 2 && coefficients[0] == 14.0 && coefficients[1] == 18.0,
      "Reference projection lost the supplied noncopyable, mutable dot.");
   success = Check(Close(NormFromDot(dot, input), std::sqrt(70.0)) && dot.calls == 3,
      "NormFromDot did not call the supplied dot instance.") && success;
   LocalProject(context, dot, basis, 2, input, coefficients);
   success = Check(dot.calls == 5 && coefficients[0] == 14.0 && coefficients[1] == 18.0,
      "LocalProject lost the supplied dot instance.") && success;
   LocalProjectEuclidean(context, basis, 2, input, coefficients);
   success = Check(coefficients[0] == 6.0 && coefficients[1] == 7.0,
      "Explicit local Euclidean projection is incorrect.") && success;
   success = Check(Close(LocalScaledEuclideanNorm(context, input), std::sqrt(29.0)),
      "Explicit local scaled Euclidean norm is incorrect.") && success;
   LocalVector extreme{3e200, 4e200, 0.0};
   success = Check(Close(LocalScaledEuclideanNorm(context, extreme) / 1e200, 5.0),
      "Local scaled Euclidean norm overflowed a representable result.") && success;
   auto negative_dot = [](const LocalVector&, const LocalVector&) { return -1.0; };
   success = Check(std::isnan(NormFromDot(negative_dot, input)),
      "NormFromDot accepted a negative self-product.") && success;
   return success;
}

} // namespace

int main(int argc, char** argv)
{
   if (argc == 2)
   {
      const std::string_view mode(argv[1]);
      Vector input(2), storage(4), empty_scratch;
      input = 1.0;
      storage = 2.0;
      auto basis = MakeVectorSlices(storage, 2, 2);
      std::array<Real, 2> coefficients{1.0, 2.0};
      EuclideanDot<DevicePolicy> dot{empty_scratch};
      if (mode == "missing-dot-scratch") { (void)dot(input, input); }
      else if (mode == "missing-norm-scratch") { (void)ScaledEuclideanNorm(dot, input); }
      else if (mode == "missing-project-scratch") { ProjectEuclidean(dot, basis, 2, input, coefficients); }
      else if (mode == "missing-combine-scratch")
      {
         HostDrivenFusedCombine<true>(basis, 2, coefficients, 1.0, input, empty_scratch);
      }
      else if (mode == "projection-size-overflow")
      {
         (void)RequiredProjectionReductionWorkspaceSize(256, std::numeric_limits<size_t>::max());
      }
      return 0;
   }

   bool success = TestCheckedSqrt();
   success = TestDotAndNorm<HostKernelConfiguration>(0) && success;
   for (size_t capacity : {size_t{1}, size_t{2}, size_t{3}, RequiredNormReductionWorkspaceSize(1031)})
   {
      success = TestDotAndNorm<DevicePolicy>(capacity) && success;
   }
   success = TestProjection<HostKernelConfiguration>(0) && success;
   for (size_t capacity : {size_t{1}, size_t{6}, RequiredProjectionReductionWorkspaceSize(1031, 3)})
   {
      success = TestProjection<DevicePolicy>(capacity) && success;
   }
   success = TestFusedCombination<false>() && success;
   success = TestFusedCombination<true>() && success;
   success = TestProjectionCoefficientOwners() && success;
   success = TestSuppliedDotAndLocalOperations() && success;
   success = Check(RequiredDotReductionWorkspaceSize(0) == 0 &&
      RequiredDotReductionWorkspaceSize(257) == 3 &&
      RequiredNormReductionWorkspaceSize(257) == 5 &&
      RequiredProjectionReductionWorkspaceSize(257, 3) == 9,
      "Reduction sizing does not match the operation partitions.") && success;
   return success ? 0 : 1;
}
