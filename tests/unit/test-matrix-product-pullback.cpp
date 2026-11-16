// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

namespace matrix_pullback_test
{
template<class T>
struct Proxy
{
   T value;
   GENDIL_HOST_DEVICE operator T() const { return value; }
   friend T operator+(Proxy, Proxy) = delete;
   friend T operator*(Proxy, Proxy) = delete;
};

template<class T, size_t Rows, size_t Columns>
struct ReadOnlyMatrix
{
   using value_type = T;
   const std::array<T, Columns>* values;
   ReadOnlyMatrix() = delete;
   GENDIL_HOST_DEVICE explicit ReadOnlyMatrix(const std::array<T, Columns>* data)
      : values(data) {}
   template<class I, class J> requires (std::is_integral_v<I> && std::is_integral_v<J>)
   GENDIL_HOST_DEVICE Proxy<T> operator()(I i, J j) const { return {values[i][j]}; }
   template<class I, class J> void operator()(I, J) = delete;
};

template<class T, size_t N>
struct ReadOnlyVector
{
   using value_type = T;
   const T* values;
   ReadOnlyVector() = delete;
   GENDIL_HOST_DEVICE explicit ReadOnlyVector(const T* data) : values(data) {}
   template<class I> requires std::is_integral_v<I>
   GENDIL_HOST_DEVICE Proxy<T> operator()(I i) const { return {values[i]}; }
   template<class I> void operator()(I) = delete;
};

// A directly evaluable value isolates pullback algebra from FE interpolation.
template<class Value, gendil::FieldShape Shape>
struct ValueExpr : gendil::FieldBase
{
   Value value;
   GENDIL_HOST_DEVICE explicit ValueExpr(const Value& input) : value(input) {}
   template<class... Args>
   GENDIL_HOST_DEVICE const Value& operator()(const Args&...) const { return value; }
};
} // namespace matrix_pullback_test

namespace gendil
{
template<class T, size_t R, size_t C>
struct static_num_rows<matrix_pullback_test::ReadOnlyMatrix<T, R, C>>
   : std::integral_constant<Integer, R> {};
template<class T, size_t R, size_t C>
struct static_num_cols<matrix_pullback_test::ReadOnlyMatrix<T, R, C>>
   : std::integral_constant<Integer, C> {};
template<class T, size_t N>
struct static_extent<matrix_pullback_test::ReadOnlyVector<T, N>>
   : std::integral_constant<Integer, N> {};
template<class Value, FieldShape Shape>
struct field_shape_impl<matrix_pullback_test::ValueExpr<Value, Shape>>
{ static constexpr FieldShape value = Shape; };
template<class Value, FieldShape Shape>
struct test_linearity<matrix_pullback_test::ValueExpr<Value, Shape>>
{
   static constexpr auto value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};
} // namespace gendil

namespace
{
using namespace gendil;
using matrix_pullback_test::ReadOnlyMatrix;
using matrix_pullback_test::ReadOnlyVector;
using matrix_pullback_test::ValueExpr;
template<size_t R, size_t C> using Matrix = std::array<std::array<Real, C>, R>;

static_assert(StaticMatrix<ReadOnlyMatrix<float, 2, 3>>);
static_assert(StaticVector<ReadOnlyVector<float, 3>>);
static_assert(!std::is_default_constructible_v<ReadOnlyMatrix<float, 2, 3>>);
static_assert(!std::is_default_constructible_v<ReadOnlyVector<float, 3>>);
using TestGradient = decltype(grad(VectorTestSpace<"v">{}));
static_assert(is_nonlinear_in_test_v<decltype(std::declval<TestGradient>() *
                                            std::declval<TestGradient>())>);

template<class Value>
GENDIL_HOST_DEVICE auto MatrixExprOf(const Value& value)
{ return ValueExpr<Value, FieldShape::Matrix>(value); }

template<class Channel, class Pullback>
void WriteSeed(const Pullback& result, Real* output)
{
   static_assert(Pullback::template Contains<Channel>());
   const auto& expression = result.template Get<Channel>();
   static_assert(field_shape_v<decltype(expression)> == FieldShape::Matrix);
   static_assert(is_test_free_v<decltype(expression)>);
   const auto value = expression(Empty{}, Empty{}, Empty{}, Empty{}, Empty{}, Empty{});
   static_assert(static_num_rows_v<decltype(value)> == 2);
   static_assert(static_num_cols_v<decltype(value)> == 3);
   for (Integer i = 0; i < 2; ++i)
      for (Integer j = 0; j < 3; ++j) output[3*i+j] = GetValue(value, i, j);
}

constexpr Integer result_size = 52;

GENDIL_HOST_DEVICE void WriteTransposeResults(Real seed, Real* output)
{
   std::array<std::array<float, 3>, 2> data{{{float(seed), -2, 3}, {4, 5, -6}}};
   const auto matrix = MatrixExprOf(ReadOnlyMatrix<float, 2, 3>(data.data()));
   const auto transposed = transpose(matrix)();
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(transposed)>,
      SerialRecursiveArray<float, 3, 2>>);
   std::array<float, 3> vector_data{float(seed), -7, 8};
   const auto vector = ValueExpr<ReadOnlyVector<float, 3>, FieldShape::Vector>(
      ReadOnlyVector<float, 3>(vector_data.data()));
   const auto row = transpose(vector)();
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(row)>,
      SerialRecursiveArray<float, 1, 3>>);
   data[0][0] = 99;
   vector_data[0] = 99;
   for (Integer i = 0; i < 3; ++i)
      for (Integer j = 0; j < 2; ++j) output[2*i+j] = transposed(i, j);
   for (Integer i = 0; i < 3; ++i) output[6+i] = row(0, i);
   output[9] = transposed(0, 0) == float(seed) && row(0, 0) == float(seed);
}

void WriteResults(Real seed, Real* output)
{
   Matrix<4, 2> a{};
   Matrix<3, 5> b{};
   Matrix<4, 3> left_seed{};
   Matrix<2, 5> right_seed{};
   Matrix<4, 5> nested_seed{};
   for (Integer i = 0; i < 4; ++i)
      for (Integer j = 0; j < 2; ++j) a[i][j] = seed + 2.0*i - j;
   for (Integer i = 0; i < 3; ++i)
      for (Integer j = 0; j < 5; ++j) b[i][j] = 1.0 + i - 2.0*j;
   for (Integer i = 0; i < 4; ++i)
      for (Integer j = 0; j < 3; ++j) left_seed[i][j] = 2.0 - i + 3.0*j;
   for (Integer i = 0; i < 2; ++i)
      for (Integer j = 0; j < 5; ++j) right_seed[i][j] = seed - 2.0*i + j;
   for (Integer i = 0; i < 4; ++i)
      for (Integer j = 0; j < 5; ++j) nested_seed[i][j] = 1.0 + 2.0*i - j;
   // Proxy factors require the adjoint's transpose to produce scalar storage.
   const auto A = MatrixExprOf(ReadOnlyMatrix<Real, 4, 2>(a.data()));
   const auto B = MatrixExprOf(ReadOnlyMatrix<Real, 3, 5>(b.data()));
   const auto L = MatrixExprOf(left_seed);
   const auto R = MatrixExprOf(right_seed);
   const auto S = MatrixExprOf(nested_seed);
   const VectorTestSpace<"v"> v;
   static_assert(is_test_linear_v<decltype(A * grad(v) * B)>);

   WriteSeed<GradientChannel>(pullback(A * grad(v), L), output);
   WriteSeed<GradientChannel>(pullback(grad(v) * B, R), output + 6);
   WriteSeed<GradientChannel>(pullback(A * grad(v) * B, S), output + 12);
   WriteSeed<GradientMinusChannel>(pullback_interior(A * minus(grad(v)), L), output + 18);
   WriteSeed<GradientPlusChannel>(pullback_interior(plus(grad(v)) * B, R), output + 24);
   const auto global = pullback_interior(A * jump(grad(v)) * B, S);
   static_assert(!decltype(global)::template Contains<ValueMinusChannel>());
   static_assert(!decltype(global)::template Contains<ValuePlusChannel>());
   WriteSeed<GradientMinusChannel>(global, output + 30);
   WriteSeed<GradientPlusChannel>(global, output + 36);

   WriteTransposeResults(seed, output + 42);
}

bool CheckResults(Real seed, const Real* actual)
{
   std::array<Real, result_size> expected{};
   for (Integer i = 0; i < 2; ++i)
      for (Integer j = 0; j < 3; ++j)
      {
         Real left = 0, right = 0, nested = 0;
         // Differentiate the explicit scalar contractions entry by entry.
         for (Integer p = 0; p < 4; ++p)
         {
            left += (seed + 2.0*p - i) * (2.0 - p + 3.0*j);
            for (Integer q = 0; q < 5; ++q)
               nested += (seed + 2.0*p - i) * (1.0 + 2.0*p - q) * (1.0 + j - 2.0*q);
         }
         for (Integer q = 0; q < 5; ++q)
            right += (seed - 2.0*i + q) * (1.0 + j - 2.0*q);
         const Integer k = 3*i+j;
         expected[k] = expected[18+k] = left;
         expected[6+k] = expected[24+k] = right;
         expected[12+k] = expected[30+k] = nested;
         expected[36+k] = -nested;
      }
   const std::array<Real, 10> transpose_expected{seed, 4, -2, 5, 3, -6, seed, -7, 8, 1};
   for (Integer i = 0; i < 10; ++i) expected[42+i] = transpose_expected[i];
   for (Integer i = 0; i < result_size; ++i)
      if (!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i]) >
          1e-12*(1+std::abs(expected[i])))
      {
         std::cerr << "Matrix pullback result " << i << ": got " << actual[i]
                   << ", expected " << expected[i] << '\n';
         return false;
      }
   return true;
}

// Traced constant factors need a side context, but no FE data for these
// algebra-only checks. The resulting gradient seed must retain its 2x3 shape.
struct FacetPoint
{
   GENDIL_HOST_DEVICE Empty MinusSide() const { return {}; }
   GENDIL_HOST_DEVICE Empty PlusSide() const { return {}; }
};
struct FacetFields { Empty minus_fields, plus_fields; };

template<class Channel, class Pullback>
bool CheckTransposeSeed(const Pullback& result, const Matrix<2, 3>& expected,
                        Real scale, const char* label)
{
   const auto& expression = result.template Get<Channel>();
   static_assert(is_test_free_v<decltype(expression)>);
   const auto value = expression(Empty{}, Empty{}, Empty{}, Empty{}, FacetPoint{}, FacetFields{});
   static_assert(static_num_rows_v<decltype(value)> == 2);
   static_assert(static_num_cols_v<decltype(value)> == 3);
   for (Integer i = 0; i < 2; ++i)
      for (Integer j = 0; j < 3; ++j)
      {
         const Real actual = GetValue(value, i, j), reference = scale*expected[i][j];
         if (!std::isfinite(actual) || std::abs(actual-reference) > 1e-12*(1+std::abs(reference)))
         {
            std::cerr << label << " at " << i << ", " << j << ": got " << actual
                      << ", expected " << reference << '\n';
            return false;
         }
      }
   return true;
}

bool CheckTransposePullbacks(Real seed)
{
   const VectorTestSpace<"v"> v;
   const Matrix<3, 2> s{{{seed, -2}, {3, 4}, {-5, 6}}};
   const Matrix<2, 3> expected{{{seed, 3, -5}, {-2, 4, 6}}};
   const auto S = MatrixExprOf(s);
   bool ok = CheckTransposeSeed<GradientChannel>(pullback(transpose(grad(v)), S),
      expected, 1, "rectangular transpose pullback");
   ok = CheckTransposeSeed<GradientChannel>(
      pullback(transpose(transpose(grad(v))), MatrixExprOf(expected)),
      expected, 1, "double transpose pullback") && ok;

   auto check_sides = [&](const auto& expr, Real minus_scale, Real plus_scale, const char* label) {
      const auto channels = pullback_interior(expr, S);
      return CheckTransposeSeed<GradientMinusChannel>(channels, expected, minus_scale, label) &&
         CheckTransposeSeed<GradientPlusChannel>(channels, expected, plus_scale, label);
   };
   ok = check_sides(transpose(jump(grad(v))), 1, -1, "transpose outside jump") && ok;
   ok = check_sides(jump(transpose(grad(v))), 1, -1, "transpose inside jump") && ok;
   ok = check_sides(transpose(average(grad(v))), .5, .5, "transpose outside average") && ok;
   ok = check_sides(average(transpose(grad(v))), .5, .5, "transpose inside average") && ok;
   ok = CheckTransposeSeed<GradientMinusChannel>(
      pullback_interior(minus(transpose(grad(v))), S),
      expected, 1, "explicit minus transpose") && ok;
   ok = CheckTransposeSeed<GradientPlusChannel>(
      pullback_interior(transpose(plus(grad(v))), S),
      expected, 1, "transpose of explicit plus") && ok;

   Matrix<4, 3> a{};
   Matrix<2, 5> b{};
   Matrix<4, 5> s_nested{};
   Matrix<2, 3> nested_expected{};
   for (Integer i = 0; i < 4; ++i)
      for (Integer j = 0; j < 3; ++j) a[i][j] = seed + 2.0*i - j;
   for (Integer i = 0; i < 2; ++i)
      for (Integer j = 0; j < 5; ++j) b[i][j] = 1.0 + i - 2.0*j;
   for (Integer i = 0; i < 4; ++i)
      for (Integer j = 0; j < 5; ++j) s_nested[i][j] = .5 + i - j;
   // Differentiate <A G^T B, S> explicitly with respect to G(i,j).
   for (Integer i = 0; i < 2; ++i)
      for (Integer j = 0; j < 3; ++j)
         for (Integer p = 0; p < 4; ++p)
            for (Integer q = 0; q < 5; ++q)
               nested_expected[i][j] += b[i][q]*s_nested[p][q]*a[p][j];
   const auto A = MatrixExprOf(ReadOnlyMatrix<Real, 4, 3>(a.data()));
   const auto B = MatrixExprOf(ReadOnlyMatrix<Real, 2, 5>(b.data()));
   const auto nested = A * transpose(grad(v)) * B;
   const auto F = MatrixExprOf(s_nested);
   ok = CheckTransposeSeed<GradientChannel>(pullback(nested, F),
      nested_expected, 1, "transpose nested in matrix products") && ok;
   const auto global = pullback_interior(jump(nested), F);
   ok = CheckTransposeSeed<GradientMinusChannel>(global, nested_expected, 1,
      "nested transpose minus channel") && ok;
   return CheckTransposeSeed<GradientPlusChannel>(global, nested_expected, -1,
      "nested transpose plus channel") && ok;
}

#ifdef GENDIL_USE_DEVICE
bool CheckDevice()
{
   using Policy = DeviceKernelConfiguration<ThreadBlockLayout<1>, 1, 4>;
   constexpr Integer count = 5;
   Vector inputs(count), results(count*10);
   auto* seeds = inputs.WriteHostData();
   for (Integer i = 0; i < count; ++i) seeds[i] = .25*(Real(i)-2.0);
   const auto* device_inputs = inputs.ReadDeviceData();
   auto* device_results = results.WriteDeviceData();
   Policy::CandidateBlockLoop(count, [=] GENDIL_DEVICE ()
   {
      if (Policy::IsActive(count))
      {
         const auto i = Policy::WorkItemIndex();
         WriteTransposeResults(device_inputs[i], device_results+10*i);
      }
   });
   GENDIL_DEVICE_SYNC;
   const auto* host = results.ReadHostData();
   bool ok = true;
   for (Integer i = 0; i < count; ++i)
   {
      const Real seed = .25*(Real(i)-2.0);
      const std::array<Real, 10> expected{seed, 4, -2, 5, 3, -6, seed, -7, 8, 1};
      for (Integer j = 0; j < 10; ++j)
         if (host[10*i+j] != expected[j])
         {
            std::cerr << "Device proxy transpose mismatch at " << i << ", " << j << '\n';
            ok = false;
         }
   }
   return ok;
}
#endif
} // namespace

int main()
{
   bool ok = true;
   for (Real seed : {Real(-.5), Real(.25), Real(1.5)})
   {
      std::array<Real, result_size> result{};
      WriteResults(seed, result.data());
      ok = CheckResults(seed, result.data()) && ok;
      ok = CheckTransposePullbacks(seed) && ok;
   }
#ifdef GENDIL_USE_DEVICE
   ok = CheckDevice() && ok;
#endif
   return ok ? 0 : 1;
}
