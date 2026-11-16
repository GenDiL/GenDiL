// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Algebra/arrayarithmetic.hpp>
#include <gendil/Algebra/matrixvectorops.hpp>
#include <gendil/gendil.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <type_traits>

namespace array_arithmetic_test
{
template<class T>
struct ScalarProxy
{
   T value;
   GENDIL_HOST_DEVICE constexpr operator T() const { return value; }
   friend T operator+(ScalarProxy, ScalarProxy) = delete;
   friend T operator-(ScalarProxy, ScalarProxy) = delete;
   friend T operator-(ScalarProxy) = delete;
   friend T operator*(T, ScalarProxy) = delete;
   friend T operator*(ScalarProxy, T) = delete;
   friend T operator*(ScalarProxy, ScalarProxy) = delete;
};

template<class T, size_t N>
struct ReadOnlyVector
{
   using value_type = T;
   const T* values;
   ReadOnlyVector() = delete;
   GENDIL_HOST_DEVICE explicit ReadOnlyVector(const T* data) : values(data) {}
   // The readable-array contract promises integral indices, not loop-tag types.
   template<class I> requires std::is_integral_v<I>
   GENDIL_HOST_DEVICE ScalarProxy<T> operator()(I i) const
   { return {values[i]}; }
};

template<class T, size_t Rows, size_t Columns>
struct ReadOnlyMatrix
{
   using value_type = T;
   const std::array<T, Columns>* values;
   ReadOnlyMatrix() = delete;
   GENDIL_HOST_DEVICE explicit ReadOnlyMatrix(const std::array<T, Columns>* data)
      : values(data) {}
   template<class I, class J>
      requires (std::is_integral_v<I> && std::is_integral_v<J>)
   GENDIL_HOST_DEVICE ScalarProxy<T> operator()(I i, J j) const
   { return {values[i][j]}; }
};
}

namespace gendil
{
template<class T, size_t N>
struct static_extent<array_arithmetic_test::ReadOnlyVector<T, N>>
   : std::integral_constant<Integer, N> {};
template<class T, size_t Rows, size_t Columns>
struct static_num_rows<array_arithmetic_test::ReadOnlyMatrix<T, Rows, Columns>>
   : std::integral_constant<Integer, Rows> {};
template<class T, size_t Rows, size_t Columns>
struct static_num_cols<array_arithmetic_test::ReadOnlyMatrix<T, Rows, Columns>>
   : std::integral_constant<Integer, Columns> {};
}

namespace
{
using namespace gendil;
using array_arithmetic_test::ReadOnlyVector;
using array_arithmetic_test::ReadOnlyMatrix;

template<class L, class R>
concept CanAdd = requires(const L& left, const R& right) { left + right; };
template<class L, class R>
concept CanSubtract = requires(const L& left, const R& right) { left - right; };
template<class L, class R>
concept CanMultiply = requires(const L& left, const R& right) { left * right; };
using ArrayVector = std::array<short, 3>;
using RecursiveVector = SerialRecursiveArray<short, 3>;
using ArrayMatrix = std::array<std::array<short, 3>, 2>;
using RecursiveMatrix = SerialRecursiveArray<short, 2, 3>;
using WrongArrayMatrix = std::array<std::array<short, 2>, 2>;
using WrongRecursiveMatrix = SerialRecursiveArray<short, 2, 2>;

static_assert(!CanAdd<ArrayVector, std::array<short, 2>>);
static_assert(!CanSubtract<ArrayVector, std::array<short, 2>>);
static_assert(!CanAdd<ArrayMatrix, WrongArrayMatrix>);
static_assert(!CanSubtract<ArrayMatrix, WrongArrayMatrix>);
static_assert(!CanAdd<ArrayMatrix, WrongRecursiveMatrix>);
static_assert(!CanAdd<WrongRecursiveMatrix, ArrayMatrix>);
static_assert(!CanSubtract<ArrayMatrix, WrongRecursiveMatrix>);
static_assert(!CanSubtract<WrongRecursiveMatrix, ArrayMatrix>);
static_assert(!CanAdd<RecursiveMatrix, WrongRecursiveMatrix>);
static_assert(!CanSubtract<RecursiveMatrix, WrongRecursiveMatrix>);
static_assert(!CanAdd<ArrayVector, RecursiveMatrix>);
static_assert(!CanSubtract<RecursiveMatrix, ArrayVector>);
static_assert(!CanAdd<ArrayVector, short>);
static_assert(!CanSubtract<short, ArrayVector>);
static_assert(StaticVector<ReadOnlyVector<float, 3>>);
static_assert(StaticMatrix<ReadOnlyMatrix<float, 2, 3>>);
static_assert(!std::is_default_constructible_v<ReadOnlyVector<float, 3>>);
static_assert(!std::is_default_constructible_v<ReadOnlyMatrix<float, 2, 3>>);
static_assert(!CanAdd<ReadOnlyVector<float, 3>, std::array<float, 2>>);
static_assert(!CanSubtract<ArrayVector, ReadOnlyVector<float, 2>>);
static_assert(!CanAdd<ReadOnlyMatrix<float, 2, 3>, ReadOnlyMatrix<float, 3, 2>>);
static_assert(!CanSubtract<ReadOnlyMatrix<float, 2, 3>, WrongRecursiveMatrix>);
static_assert(!CanAdd<ReadOnlyVector<float, 3>, ReadOnlyMatrix<float, 2, 3>>);
static_assert(!CanSubtract<ReadOnlyMatrix<float, 2, 3>, ReadOnlyVector<float, 3>>);
static_assert(!CanMultiply<ReadOnlyVector<float, 3>, ReadOnlyVector<float, 3>>);
static_assert(!CanMultiply<ArrayMatrix, ArrayMatrix>);
static_assert(!CanMultiply<ArrayMatrix, RecursiveMatrix>);
static_assert(!CanMultiply<RecursiveMatrix, ArrayMatrix>);
static_assert(!CanMultiply<RecursiveMatrix, RecursiveMatrix>);
static_assert(!CanMultiply<ReadOnlyMatrix<float, 2, 3>, WrongArrayMatrix>);
static_assert(!CanMultiply<ReadOnlyMatrix<float, 2, 3>, ReadOnlyMatrix<float, 2, 4>>);
static_assert(!CanMultiply<ArrayVector, WrongArrayMatrix>);
static_assert(CanMultiply<ArrayMatrix, std::array<std::array<short, 4>, 3>>);
static_assert(CanAdd<float[3], ReadOnlyVector<float, 3>>);
static_assert(CanAdd<ReadOnlyVector<float, 3>, float[3]>);
static_assert(CanSubtract<float[3], ArrayVector>);
static_assert(CanSubtract<RecursiveVector, float[3]>);
static_assert(!CanAdd<float[3], float[3]>);
// Raw arrays use built-in pointer subtraction; no array operator participates.
// Probe the expression rather than forming an invalid raw/raw operator signature.
static_assert(std::is_same_v<decltype(std::declval<const float(&)[3]>() -
                                    std::declval<const float(&)[3]>()), std::ptrdiff_t>);

template<class T, size_t N>
GENDIL_HOST_DEVICE
auto Recursive(const std::array<T, N>& array)
{
   SerialRecursiveArray<T, N> result{};
   for (size_t i = 0; i < N; ++i) { result(i) = array[i]; }
   return result;
}

template<class T, size_t Rows, size_t Columns>
GENDIL_HOST_DEVICE
auto Recursive(const std::array<std::array<T, Columns>, Rows>& array)
{
   SerialRecursiveArray<T, Rows, Columns> result{};
   for (size_t i = 0; i < Rows; ++i)
      for (size_t j = 0; j < Columns; ++j) { result(i, j) = array[i][j]; }
   return result;
}

template<class Value, class T, size_t N>
GENDIL_HOST_DEVICE
bool Equal(const Value& value, const std::array<T, N>& expected)
{
   for (size_t i = 0; i < N; ++i)
      if (GetValue(value, i) != expected[i]) { return false; }
   return true;
}

template<class Value, class T, size_t Rows, size_t Columns>
GENDIL_HOST_DEVICE
bool Equal(const Value& value, const std::array<std::array<T, Columns>, Rows>& expected)
{
   for (size_t i = 0; i < Rows; ++i)
      for (size_t j = 0; j < Columns; ++j)
         if (GetValue(value, i, j) != expected[i][j]) { return false; }
   return true;
}

template<class Result, class Expected>
constexpr bool HasType = std::is_same_v<std::remove_cvref_t<Result>, Expected> &&
                         !std::is_reference_v<Result>;

constexpr bool CheckConstexprStandardArrays()
{
   const std::array<short, 2> a{1, -3}, b{2, 4};
   const auto sum = a+b, difference = a-b, negative = NegateValue(a);
   const auto left = .5*a, right = a*.5;
   const std::array<std::array<short, 2>, 2> matrix{{{1, -3}, {4, 2}}};
   const auto matrix_sum = matrix+matrix, matrix_difference = matrix-matrix;
   const auto matrix_negative = NegateValue(matrix);
   return sum[0] == 3 && sum[1] == 1 && difference[0] == -1 &&
      difference[1] == -7 && negative[0] == -1 && negative[1] == 3 &&
      left[0] == .5 && left[1] == -1.5 && right == left &&
      matrix_sum[1][0] == 8 && matrix_difference[0][1] == 0 &&
      matrix_negative[0][1] == 3;
}
static_assert(CheckConstexprStandardArrays());

template<class Result, class L, class R, class Expected>
GENDIL_HOST_DEVICE
bool CheckBinary(const L& left, const R& right, const Expected& sum,
                 const Expected& difference, const Expected& reverse)
{
   static_assert(HasType<decltype(left+right), Result>);
   static_assert(HasType<decltype(right+left), Result>);
   static_assert(HasType<decltype(left-right), Result>);
   static_assert(HasType<decltype(right-left), Result>);
   return Equal(left+right, sum) && Equal(right+left, sum) &&
      Equal(left-right, difference) && Equal(right-left, reverse);
}

GENDIL_HOST_DEVICE
bool CheckProxyVectors(float seed)
{
   std::array<float, 3> a{seed, -2.f, 3.f};
   const std::array<float, 3> b{.5f, 4.f, -1.5f};
   const ReadOnlyVector<float, 3> left(a.data()), right(b.data());
   const auto recursive = Recursive(b);
   const std::array<float, 3> sum{seed+.5f, 2.f, 1.5f};
   const std::array<float, 3> difference{seed-.5f, -6.f, 4.5f};
   const std::array<float, 3> reverse{.5f-seed, 6.f, -4.5f};
   using Result = SerialRecursiveArray<float, 3>;
   bool ok = CheckBinary<Result>(left, b, sum, difference, reverse) &&
      CheckBinary<Result>(left, recursive, sum, difference, reverse) &&
      CheckBinary<Result>(left, right, sum, difference, reverse);
   const float raw[3]{seed, -2.f, 3.f};
   ok = CheckBinary<Result>(raw, right, sum, difference, reverse) &&
      CheckBinary<Result>(raw, b, sum, difference, reverse) &&
      CheckBinary<Result>(raw, recursive, sum, difference, reverse) && ok;
   const std::array<double, 3> wide{.5, 4., -1.5};
   ok = CheckBinary<SerialRecursiveArray<double, 3>>(
      left, wide, sum, difference, reverse) && ok;

   const auto scaled_left = .5*left, scaled_right = left*.5;
   const auto negative = NegateValue(left);
   const auto average = details::AverageValues(left, right);
   const auto saved_sum = left+right;
   static_assert(HasType<decltype(scaled_left), SerialRecursiveArray<double, 3>>);
   static_assert(HasType<decltype(scaled_right), SerialRecursiveArray<double, 3>>);
   static_assert(HasType<decltype(negative), Result>);
   static_assert(HasType<decltype(average), Result>);
   const std::array<double, 3> scaled{.5*seed, -1., 1.5};
   const std::array<float, 3> negated{-seed, 2.f, -3.f};
   const std::array<float, 3> mean{.5f*seed+.25f, 1.f, .75f};
   ok = Equal(a, std::array<float, 3>{seed, -2.f, 3.f}) &&
      Equal(b, std::array<float, 3>{.5f, 4.f, -1.5f}) && ok;
   a[0] = 99.f; // A returned value must not retain a view into this source.
   return Equal(saved_sum, sum) && Equal(scaled_left, scaled) &&
      Equal(scaled_right, scaled) && Equal(negative, negated) && Equal(average, mean) && ok;
}

GENDIL_HOST_DEVICE
bool CheckProxyMatrices(float seed)
{
   std::array<std::array<float, 3>, 2> a{{{seed, -2.f, 3.f}, {4.f, 5.f, -6.f}}};
   const std::array<std::array<float, 3>, 2> b{{{.5f, 4.f, -1.5f}, {2.f, -3.f, 8.f}}};
   const ReadOnlyMatrix<float, 2, 3> left(a.data()), right(b.data());
   const auto recursive = Recursive(b);
   const std::array<std::array<float, 3>, 2> sum{{{seed+.5f, 2.f, 1.5f}, {6.f, 2.f, 2.f}}};
   const std::array<std::array<float, 3>, 2> difference{{{seed-.5f, -6.f, 4.5f}, {2.f, 8.f, -14.f}}};
   const std::array<std::array<float, 3>, 2> reverse{{{.5f-seed, 6.f, -4.5f}, {-2.f, -8.f, 14.f}}};
   using Result = SerialRecursiveArray<float, 2, 3>;
   bool ok = CheckBinary<Result>(left, b, sum, difference, reverse) &&
      CheckBinary<Result>(left, recursive, sum, difference, reverse) &&
      CheckBinary<Result>(left, right, sum, difference, reverse);
   const auto scaled_left = .5*left, scaled_right = left*.5;
   const auto negative = NegateValue(left);
   const auto average = details::AverageValues(left, right);
   const auto saved_difference = left-right;
   static_assert(HasType<decltype(scaled_left), SerialRecursiveArray<double, 2, 3>>);
   static_assert(HasType<decltype(scaled_right), SerialRecursiveArray<double, 2, 3>>);
   static_assert(HasType<decltype(negative), Result>);
   static_assert(HasType<decltype(average), Result>);
   const std::array<std::array<double, 3>, 2> scaled{{{.5*seed, -1., 1.5}, {2., 2.5, -3.}}};
   const std::array<std::array<float, 3>, 2> negated{{{-seed, 2.f, -3.f}, {-4.f, -5.f, 6.f}}};
   const std::array<std::array<float, 3>, 2> mean{{{.5f*seed+.25f, 1.f, .75f}, {3.f, 1.f, 1.f}}};
   ok = Equal(a, std::array<std::array<float, 3>, 2>{{{seed, -2.f, 3.f}, {4.f, 5.f, -6.f}}}) &&
      Equal(recursive, b) && ok;
   a[0][0] = 99.f;
   return Equal(saved_difference, difference) && Equal(scaled_left, scaled) &&
      Equal(scaled_right, scaled) && Equal(negative, negated) && Equal(average, mean) && ok;
}

GENDIL_HOST_DEVICE
bool CheckVectors()
{
   const ArrayVector a{1, -2, 3};
   const std::array<double, 3> b{.5, 4., -1.5};
   const auto ra = Recursive(a);
   const auto rb = Recursive(b);
   const std::array<double, 3> sum{1.5, 2., 1.5};
   const std::array<double, 3> difference{.5, -6., 4.5};
   const std::array<double, 3> reverse{-.5, 6., -4.5};
   static_assert(HasType<decltype(a + a), std::array<int, 3>>);
   static_assert(HasType<decltype(a - a), std::array<int, 3>>);
   static_assert(HasType<decltype(a + b), std::array<double, 3>>);
   static_assert(HasType<decltype(a - b), std::array<double, 3>>);
   static_assert(HasType<decltype(a + rb), SerialRecursiveArray<double, 3>>);
   static_assert(HasType<decltype(ra + b), SerialRecursiveArray<double, 3>>);
   static_assert(HasType<decltype(a - rb), SerialRecursiveArray<double, 3>>);
   static_assert(HasType<decltype(ra - b), SerialRecursiveArray<double, 3>>);
   static_assert(HasType<decltype(ra - rb), SerialRecursiveArray<double, 3>>);
   static_assert(HasType<decltype(ra + ra), RecursiveVector>);
   static_assert(HasType<decltype(ra - ra), RecursiveVector>);
   bool ok = Equal(a+b, sum) && Equal(a+rb, sum) && Equal(ra+b, sum) &&
      Equal(ra+rb, sum) && Equal(b+a, sum) && Equal(rb+a, sum) &&
      Equal(b+ra, sum) && Equal(rb+ra, sum);
   ok = Equal(a-b, difference) && Equal(a-rb, difference) &&
      Equal(ra-b, difference) && Equal(ra-rb, difference) &&
      Equal(b-a, reverse) && Equal(rb-a, reverse) &&
      Equal(b-ra, reverse) && Equal(rb-ra, reverse) && ok;
   auto owned_array = a+b;
   auto owned_recursive = a+rb;
   owned_array[0] = 99.;
   owned_recursive(1) = 99.;
   return Equal(a, ArrayVector{1, -2, 3}) && Equal(ra, a) &&
      Equal(b, std::array<double, 3>{.5, 4., -1.5}) && Equal(rb, b) &&
      owned_array[0] == 99. && owned_recursive(1) == 99. && ok;
}

GENDIL_HOST_DEVICE
bool CheckMatrices()
{
   const ArrayMatrix a{{{1, -2, 3}, {4, 5, -6}}};
   const std::array<std::array<float, 3>, 2> b{{{.5f, 4.f, -1.5f}, {2.f, -3.f, 8.f}}};
   const auto ra = Recursive(a);
   const auto rb = Recursive(b);
   const std::array<std::array<float, 3>, 2> sum{{{1.5f, 2.f, 1.5f}, {6.f, 2.f, 2.f}}};
   const std::array<std::array<float, 3>, 2> difference{{{.5f, -6.f, 4.5f}, {2.f, 8.f, -14.f}}};
   const std::array<std::array<float, 3>, 2> reverse{{{-.5f, 6.f, -4.5f}, {-2.f, -8.f, 14.f}}};
   static_assert(HasType<decltype(a+a), std::array<std::array<int, 3>, 2>>);
   static_assert(HasType<decltype(a-a), std::array<std::array<int, 3>, 2>>);
   static_assert(HasType<decltype(a+b), std::array<std::array<float, 3>, 2>>);
   static_assert(HasType<decltype(a-b), std::array<std::array<float, 3>, 2>>);
   static_assert(HasType<decltype(a-rb), SerialRecursiveArray<float, 2, 3>>);
   static_assert(HasType<decltype(ra-b), SerialRecursiveArray<float, 2, 3>>);
   static_assert(HasType<decltype(ra-rb), SerialRecursiveArray<float, 2, 3>>);
   bool ok = Equal(a+b, sum) && Equal(a+rb, sum) && Equal(ra+b, sum) &&
      Equal(ra+rb, sum) && Equal(b+a, sum) && Equal(rb+a, sum) &&
      Equal(b+ra, sum) && Equal(rb+ra, sum);
   ok = Equal(a-b, difference) && Equal(a-rb, difference) &&
      Equal(ra-b, difference) && Equal(ra-rb, difference) &&
      Equal(b-a, reverse) && Equal(rb-a, reverse) &&
      Equal(b-ra, reverse) && Equal(rb-ra, reverse) && ok;
   auto owned = a-rb;
   owned(1, 2) = 99.f;
   return Equal(a, ArrayMatrix{{{1, -2, 3}, {4, 5, -6}}}) &&
      Equal(ra, a) && Equal(rb, b) && owned(1, 2) == 99.f && ok;
}

template<class Left, class Right>
GENDIL_HOST_DEVICE
bool CheckMatrixProductPair(const Left& left, const Right& right, double seed)
{
   const std::array<std::array<double, 2>, 2> expected{
      {{.5*seed-12.5, 2*seed+30.}, {31., -55.}}};
   const std::array<std::array<double, 3>, 3> reverse{
      {{.5*seed+8., 9., -10.5}, {4*seed-12., -23., 30.},
       {-1.5*seed+32., 43., -52.5}}};
   static_assert(HasType<decltype(left*right), SerialRecursiveArray<double, 2, 2>>);
   static_assert(HasType<decltype(right*left), SerialRecursiveArray<double, 3, 3>>);
   return Equal(left*right, expected) && Equal(right*left, reverse);
}

GENDIL_HOST_DEVICE
bool CheckMatrixProducts(float seed)
{
   std::array<std::array<float, 3>, 2> a{{{seed, -2.f, 3.f}, {4.f, 5.f, -6.f}}};
   std::array<std::array<double, 2>, 3> b{{{.5, 2.}, {4., -3.}, {-1.5, 8.}}};
   const auto ra = Recursive(a);
   const auto rb = Recursive(b);
   const ReadOnlyMatrix<float, 2, 3> pa(a.data());
   const ReadOnlyMatrix<double, 3, 2> pb(b.data());
   bool ok = CheckMatrixProductPair(a, b, seed) && CheckMatrixProductPair(a, rb, seed) &&
      CheckMatrixProductPair(ra, b, seed) && CheckMatrixProductPair(ra, rb, seed) &&
      CheckMatrixProductPair(pa, b, seed) && CheckMatrixProductPair(pa, rb, seed) &&
      CheckMatrixProductPair(pa, pb, seed) && CheckMatrixProductPair(a, pb, seed) &&
      CheckMatrixProductPair(ra, pb, seed);

   // Same-entry recursive products are new algebraic contractions; scalar
   // promotion applies even though native recursive addition retains short.
   const std::array<std::array<short, 2>, 1> small_left{{{200, -200}}};
   const std::array<std::array<short, 1>, 2> small_right{{{200}, {-200}}};
   const auto narrow_left = Recursive(small_left);
   const auto narrow_right = Recursive(small_right);
   const auto promoted = narrow_left*narrow_right;
   static_assert(HasType<decltype(promoted), SerialRecursiveArray<int, 1, 1>>);
   static_assert(HasType<decltype(small_left*small_right), SerialRecursiveArray<int, 1, 1>>);
   ok = promoted(0, 0) == 80000 && (small_left*small_right)(0, 0) == 80000 && ok;

   const std::array<std::array<short, 0>, 2> empty_left{};
   const std::array<std::array<short, 3>, 0> empty_right{};
   const std::array<std::array<int, 3>, 2> zeros{};
   ok = Equal(empty_left*empty_right, zeros) && ok;

   const auto owned = pa*pb;
   const std::array<std::array<double, 2>, 2> expected{
      {{.5*seed-12.5, 2*seed+30.}, {31., -55.}}};
   ok = Equal(ra, a) && Equal(rb, b) && ok;
   a[0][0] = 99.f;
   b[0][0] = -99.;
   return Equal(owned, expected) && ok;
}

GENDIL_HOST_DEVICE
bool CheckScaling()
{
   const ArrayVector a{1, -2, 3};
   const auto ra = Recursive(a);
   const std::array<double, 3> expected{.5, -1., 1.5};
   static_assert(HasType<decltype(.5*a), std::array<double, 3>>);
   static_assert(HasType<decltype(a*.5), std::array<double, 3>>);
   static_assert(HasType<decltype(.5*ra), SerialRecursiveArray<double, 3>>);
   static_assert(HasType<decltype(ra*.5), SerialRecursiveArray<double, 3>>);
   bool ok = Equal(.5*a, expected) && Equal(a*.5, expected) &&
      Equal(.5*ra, expected) && Equal(ra*.5, expected);
   const ArrayMatrix m{{{1, -2, 3}, {4, 5, -6}}};
   const auto rm = Recursive(m);
   const std::array<std::array<double, 3>, 2> matrix_expected{{{.5, -1., 1.5}, {2., 2.5, -3.}}};
   static_assert(HasType<decltype(.5*m), SerialRecursiveArray<double, 2, 3>>);
   static_assert(HasType<decltype(m*.5), SerialRecursiveArray<double, 2, 3>>);
   static_assert(HasType<decltype(.5*rm), SerialRecursiveArray<double, 2, 3>>);
   static_assert(HasType<decltype(rm*.5), SerialRecursiveArray<double, 2, 3>>);
   ok = Equal(.5*m, matrix_expected) && Equal(m*.5, matrix_expected) &&
      Equal(.5*rm, matrix_expected) && Equal(rm*.5, matrix_expected) && ok;
   auto owned = ra*.5;
   owned(0) = 99.;
   return Equal(ra, a) && Equal(rm, m) && owned(0) == 99. && ok;
}

GENDIL_HOST_DEVICE
bool CheckNegation()
{
   const ArrayVector a{1, -2, 3};
   const auto ra = Recursive(a);
   const std::array<int, 3> expected{-1, 2, -3};
   static_assert(HasType<decltype(NegateValue(a)), std::array<int, 3>>);
   // Recursive-array native negation retains its existing entry type.
   static_assert(HasType<decltype(NegateValue(ra)), RecursiveVector>);
   const ArrayMatrix m{{{1, -2, 3}, {4, 5, -6}}};
   const auto rm = Recursive(m);
   const std::array<std::array<int, 3>, 2> matrix_expected{{{-1, 2, -3}, {-4, -5, 6}}};
   static_assert(HasType<decltype(NegateValue(m)), std::array<std::array<int, 3>, 2>>);
   static_assert(HasType<decltype(NegateValue(rm)), RecursiveMatrix>);
   return Equal(NegateValue(a), expected) && Equal(NegateValue(ra), expected) &&
      Equal(NegateValue(m), matrix_expected) && Equal(NegateValue(rm), matrix_expected) &&
      Equal(ra, a) && Equal(rm, m);
}

template<class T>
GENDIL_HOST_DEVICE
bool CheckAveragesOfType()
{
   using ResultScalar = std::conditional_t<std::is_floating_point_v<T>, T, Real>;
   const T left_scalar = 1, right_scalar = 2;
   const auto scalar = details::AverageValues(left_scalar, right_scalar);
   static_assert(HasType<decltype(details::AverageValues(left_scalar, right_scalar)), ResultScalar>);

   const std::array<T, 3> a{1, -2, 3}, b{2, 1, -4};
   const auto ra = Recursive(a), rb = Recursive(b);
   const std::array<ResultScalar, 3> expected{1.5, -.5, -.5};
   const auto vector = details::AverageValues(a, b);
   const auto recursive_vector = details::AverageValues(ra, rb);
   static_assert(HasType<decltype(vector), std::array<ResultScalar, 3>>);
   static_assert(HasType<decltype(recursive_vector), SerialRecursiveArray<ResultScalar, 3>>);

   const std::array<std::array<T, 3>, 2> m{{{1, -2, 3}, {4, 5, -6}}};
   const std::array<std::array<T, 3>, 2> n{{{2, 1, -4}, {-3, -6, 1}}};
   const auto rm = Recursive(m), rn = Recursive(n);
   const std::array<std::array<ResultScalar, 3>, 2> matrix_expected{
      {{1.5, -.5, -.5}, {.5, -.5, -2.5}}};
   const auto matrix = details::AverageValues(m, n);
   const auto recursive_matrix = details::AverageValues(rm, rn);
   static_assert(HasType<decltype(matrix), SerialRecursiveArray<ResultScalar, 2, 3>>);
   static_assert(HasType<decltype(recursive_matrix), SerialRecursiveArray<ResultScalar, 2, 3>>);
   return scalar == ResultScalar(1.5) && Equal(vector, expected) &&
      Equal(recursive_vector, expected) && Equal(matrix, matrix_expected) &&
      Equal(recursive_matrix, matrix_expected) &&
      Equal(ra, a) && Equal(rb, b) && Equal(rm, m) && Equal(rn, n);
}

GENDIL_HOST_DEVICE
bool CheckAverages()
{
   return CheckAveragesOfType<int>() && CheckAveragesOfType<float>() &&
          CheckAveragesOfType<double>();
}

struct ProductProxy { int value; };
struct CustomScalar
{
   int value{};
   GENDIL_HOST_DEVICE CustomScalar& operator=(ProductProxy product)
   { value = product.value; return *this; }
   GENDIL_HOST_DEVICE friend CustomScalar operator+(CustomScalar a, CustomScalar b)
   { return {a.value+b.value}; }
   GENDIL_HOST_DEVICE friend CustomScalar operator-(CustomScalar a, CustomScalar b)
   { return {a.value-b.value}; }
};
struct CustomFactor { int value; };
GENDIL_HOST_DEVICE ProductProxy operator*(CustomFactor a, CustomScalar b)
{ return {a.value*b.value}; }
GENDIL_HOST_DEVICE ProductProxy operator*(CustomScalar a, CustomFactor b)
{ return {a.value*b.value}; }

GENDIL_HOST_DEVICE
bool CheckNativeCompatibility()
{
   using Tensor = std::array<std::array<std::array<short, 2>, 1>, 2>;
   Tensor a{}, b{};
   a[0][0] = {1, -2}; a[1][0] = {3, 4};
   b[0][0] = {2, 3}; b[1][0] = {4, 5};
   SerialRecursiveArray<short, 2, 1, 2> ra{}, rb{};
   for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < 2; ++j)
      { ra(i, 0, j) = a[i][0][j]; rb(i, 0, j) = b[i][0][j]; }
   const auto sum = a+b, difference = a-b;
   const auto mixed_sum = a+rb, reverse_sum = ra+b;
   const auto mixed_difference = a-rb, reverse_difference = rb-a;
   const auto native_sum = ra+rb, native_difference = ra-rb;
   static_assert(HasType<decltype(sum), std::array<std::array<std::array<int, 2>, 1>, 2>>);
   static_assert(HasType<decltype(mixed_sum), SerialRecursiveArray<int, 2, 1, 2>>);
   static_assert(HasType<decltype(native_sum), SerialRecursiveArray<short, 2, 1, 2>>);
   const int expected_sum[2][2]{{3, 1}, {7, 9}};
   const int expected_difference[2][2]{{-1, -5}, {-1, -1}};
   bool ok = true;
   for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < 2; ++j)
         ok = sum[i][0][j] == expected_sum[i][j] &&
            difference[i][0][j] == expected_difference[i][j] &&
            mixed_sum(i, 0, j) == expected_sum[i][j] &&
            reverse_sum(i, 0, j) == expected_sum[i][j] &&
            mixed_difference(i, 0, j) == expected_difference[i][j] &&
            reverse_difference(i, 0, j) == -expected_difference[i][j] &&
            native_sum(i, 0, j) == expected_sum[i][j] &&
            native_difference(i, 0, j) == expected_difference[i][j] && ok;

   const std::array<CustomScalar, 2> c{{{2}, {-3}}}, d{{{4}, {5}}};
   const auto rc = Recursive(c), rd = Recursive(d);
   const auto custom_sum = c+d, custom_difference = c-d;
   const auto custom_mixed = c+rd, custom_reverse = rd-c;
   const auto left_scaled = CustomFactor{3}*rc, right_scaled = rc*CustomFactor{3};
   static_assert(HasType<decltype(custom_sum), std::array<CustomScalar, 2>>);
   static_assert(HasType<decltype(custom_mixed), SerialRecursiveArray<CustomScalar, 2>>);
   static_assert(HasType<decltype(left_scaled), SerialRecursiveArray<CustomScalar, 2>>);
   ok = custom_sum[0].value == 6 && custom_sum[1].value == 2 &&
      custom_difference[0].value == -2 && custom_difference[1].value == -8 &&
      custom_mixed(0).value == 6 && custom_mixed(1).value == 2 &&
      custom_reverse(0).value == 2 && custom_reverse(1).value == 8 &&
      left_scaled(0).value == 6 && left_scaled(1).value == -9 &&
      right_scaled(0).value == 6 && right_scaled(1).value == -9 && ok;
   ok = (c[0]*CustomFactor{3}).value == 6 && ok;
   SerialRecursiveArray<short> scalar{};
   scalar() = 3;
   const auto native_negative = NegateValue(scalar);
   const auto scaled_scalar = .5*scalar;
   static_assert(HasType<decltype(native_negative), SerialRecursiveArray<short>>);
   static_assert(HasType<decltype(scaled_scalar), SerialRecursiveArray<double>>);
   return native_negative() == -3 && scaled_scalar() == 1.5 && ok;
}

// Materialize numeric results from runtime input, so the device smoke test
// checks transferred arithmetic values as well as the boolean contract tests.
constexpr size_t runtime_result_size = 13;
GENDIL_HOST_DEVICE
void WriteRuntimeProxyResults(float seed, Real* output)
{
   using gendil::operator*;
   const std::array<float, 3> a{seed, -2.f, 3.f}, b{.5f, 4.f, -1.5f};
   const ReadOnlyVector<float, 3> vector(a.data());
   const auto vector_result = (vector+b)*.5;
   const std::array<std::array<float, 3>, 2> m{{{seed, -2.f, 3.f}, {4.f, 5.f, -6.f}}};
   const std::array<std::array<float, 3>, 2> n{{{.5f, 4.f, -1.5f}, {2.f, -3.f, 8.f}}};
   const ReadOnlyMatrix<float, 2, 3> matrix(m.data());
   const auto matrix_result = NegateValue(matrix-Recursive(n))*.5f;
   for (size_t i = 0; i < 3; ++i) { output[i] = vector_result(i); }
   for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < 3; ++j) { output[3+3*i+j] = matrix_result(i, j); }
   const std::array<std::array<double, 2>, 3> bmatrix{{{.5, 2.}, {4., -3.}, {-1.5, 8.}}};
   const ReadOnlyMatrix<double, 3, 2> right(bmatrix.data());
   const auto product = matrix*right;
   for (size_t i = 0; i < 2; ++i)
      for (size_t j = 0; j < 2; ++j) { output[9+2*i+j] = product(i, j); }
}

bool CheckRuntimeProxyResults(Real seed, const Real* result)
{
   const std::array<Real, runtime_result_size> expected{
      .5*seed+.25, 1., .75, .25-.5*seed, 3., -2.25, -1., -4., 7.,
      .5*seed-12.5, 2*seed+30., 31., -55.};
   for (size_t j = 0; j < runtime_result_size; ++j)
      if (result[j] != expected[j]) { return false; }
   return true;
}

#ifdef GENDIL_USE_DEVICE
bool CheckDevice()
{
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<1>, 1, 4>;
   Vector results(8), inputs(8), numeric_results(8*runtime_result_size);
   auto* host_inputs = inputs.WriteHostData();
   for (size_t i = 0; i < 8; ++i) { host_inputs[i] = .25*(i+1); }
   const auto* seeds = inputs.ReadDeviceData();
   auto* values = results.WriteDeviceData();
   auto* numeric = numeric_results.WriteDeviceData();
   Policy::CandidateBlockLoop(8, [=] GENDIL_DEVICE ()
   {
      const auto i = Policy::WorkItemIndex();
      if (Policy::IsActive(8))
      {
         if (i == 0) { values[i] = CheckVectors(); }
         if (i == 1) { values[i] = CheckMatrices(); }
         if (i == 2) { values[i] = CheckScaling(); }
         if (i == 3) { values[i] = CheckNegation(); }
         if (i == 4) { values[i] = CheckAverages(); }
         if (i == 5) { values[i] = CheckProxyVectors(static_cast<float>(seeds[i])); }
         if (i == 6) { values[i] = CheckProxyMatrices(static_cast<float>(seeds[i])) &&
            CheckMatrixProducts(static_cast<float>(seeds[i])); }
         if (i == 7) { values[i] = CheckNativeCompatibility(); }
         WriteRuntimeProxyResults(static_cast<float>(seeds[i]), numeric+runtime_result_size*i);
      }
   });
   GENDIL_DEVICE_SYNC;
   const auto* result = results.ReadHostData();
   const auto* numeric_result = numeric_results.ReadHostData();
   for (size_t i = 0; i < 8; ++i)
   {
      if (result[i] != 1.) { return false; }
      const Real seed = .25*(i+1);
      if (!CheckRuntimeProxyResults(seed, numeric_result+runtime_result_size*i)) { return false; }
   }
   return true;
}
#endif
} // namespace

int main()
{
   const bool vectors = CheckVectors(), matrices = CheckMatrices();
   const bool scaling = CheckScaling(), negation = CheckNegation();
   const bool averages = CheckAverages() && CheckAveragesOfType<long double>();
   const bool proxies = CheckProxyVectors(1.25f) && CheckProxyVectors(-.75f) &&
      CheckProxyMatrices(1.25f) && CheckProxyMatrices(-.75f);
   const bool compatibility = CheckNativeCompatibility();
   const bool products = CheckMatrixProducts(1.25f) && CheckMatrixProducts(-.75f);
   std::array<Real, runtime_result_size> runtime_results{};
   WriteRuntimeProxyResults(1.25f, runtime_results.data());
   const bool runtime_values = CheckRuntimeProxyResults(1.25, runtime_results.data());
   if (!vectors) { std::cerr << "Vector array arithmetic failed.\n"; }
   if (!matrices) { std::cerr << "Matrix array arithmetic failed.\n"; }
   if (!scaling) { std::cerr << "Promoted scalar array multiplication failed.\n"; }
   if (!negation) { std::cerr << "Array negation failed.\n"; }
   if (!averages) { std::cerr << "Array average value or scalar type failed.\n"; }
   if (!proxies) { std::cerr << "Read-only proxy arithmetic failed.\n"; }
   if (!compatibility) { std::cerr << "Higher-rank or custom-scalar arithmetic failed.\n"; }
   if (!products) { std::cerr << "Static matrix multiplication failed.\n"; }
   if (!runtime_values) { std::cerr << "Runtime proxy arithmetic values failed.\n"; }
   bool ok = vectors && matrices && scaling && negation && averages && proxies &&
      compatibility && products && runtime_values;
#ifdef GENDIL_USE_DEVICE
   const bool device = CheckDevice();
   if (!device) { std::cerr << "Device array arithmetic failed.\n"; }
   ok = device && ok;
#endif
   return ok ? 0 : 1;
}
