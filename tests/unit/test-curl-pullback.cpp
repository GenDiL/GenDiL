// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#include <gendil/gendil.hpp>
#include <gendil/FiniteElementMethod/WeakForm/curl.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace curl_numeric_test
{
struct StaticMatrix
{
   std::array<std::array<long double, 3>, 3> values;
   const long double& operator()(gendil::Integer i, gendil::Integer j) const
   { return values[i][j]; }
};
struct ScalarProxy
{
   float value;
   constexpr operator float() const { return value; }
   friend float operator-(ScalarProxy, ScalarProxy) = delete;
};
struct ProxyMatrix
{
   using value_type = float;
   std::array<std::array<float, 3>, 3> values;
   constexpr ScalarProxy operator()(gendil::Integer i, gendil::Integer j) const
   { return {values[i][j]}; }
};
}

namespace gendil
{
template<> struct static_num_rows<curl_numeric_test::StaticMatrix>
   : std::integral_constant<Integer, 3> {};
template<> struct static_num_cols<curl_numeric_test::StaticMatrix>
   : std::integral_constant<Integer, 3> {};
template<> struct static_num_rows<curl_numeric_test::ProxyMatrix>
   : std::integral_constant<Integer, 3> {};
template<> struct static_num_cols<curl_numeric_test::ProxyMatrix>
   : std::integral_constant<Integer, 3> {};
}

namespace
{
using namespace gendil;
template<size_t Dim> using Coordinates = std::array<Real, Dim>;
template<size_t Dim> using Matrix = std::array<Coordinates<Dim>, Dim>;
template<size_t Components, size_t Dim>
using Gradient = std::array<Coordinates<Dim>, Components>;

struct TestKernel
{
   template<Integer Dim> using register_dimensions = std::make_index_sequence<Dim>;
};

template<class Expression, class Original>
constexpr bool ContainsOriginalExpression()
{
   using E = std::remove_cvref_t<Expression>;
   if constexpr (std::is_same_v<E, Original>) return true;
   else if constexpr (requires(const E& e) { e.lhs; e.rhs; })
      return ContainsOriginalExpression<decltype(std::declval<E>().lhs), Original>() ||
             ContainsOriginalExpression<decltype(std::declval<E>().rhs), Original>();
   else if constexpr (requires(const E& e) { e.expr; })
      return ContainsOriginalExpression<decltype(std::declval<E>().expr), Original>();
   else if constexpr (requires(const E& e) { e.seed; })
      return ContainsOriginalExpression<decltype(std::declval<E>().seed), Original>();
   else return false;
}

template<size_t Dim>
struct GradientTensor
{
   Coordinates<Dim> values{};
   template<class... Indices>
   Real& operator()(Indices... indices)
   {
      const std::array<Integer, sizeof...(Indices)> index{Integer(indices)...};
      return values[index.back()];
   }
   template<class... Indices>
   Real operator()(Indices... indices) const
   {
      const std::array<Integer, sizeof...(Indices)> index{Integer(indices)...};
      return values[index.back()];
   }
};

template<size_t Components, size_t Dim>
auto Field(const Gradient<Components, Dim>& gradient)
{
   auto tensors = [&]<size_t... I>(std::index_sequence<I...>)
   { return std::tuple{GradientTensor<Dim>{gradient[I]}...}; }
      (std::make_index_sequence<Components>{});
   return InterpolatedField<Empty, decltype(tensors)>{{}, tensors};
}

bool Close(Real actual, Real expected, const char* label)
{
   if (!std::isfinite(actual) || !std::isfinite(expected) ||
       std::abs(actual-expected) > 1e-12*(1+std::abs(expected)))
   {
      std::cerr << label << ": got " << actual << ", expected " << expected << '\n';
      return false;
   }
   return true;
}

template<class Gradient, class T, size_t Size>
bool CheckNumericCurl(const Gradient& gradient, const std::array<T, Size>& expected,
                      const char* label)
{
   const auto result = weak_form_detail::CurlFromGradient{}(gradient);
   static_assert(std::is_same_v<vector_value_t<decltype(result)>, T>);
   static_assert(static_extent_v<decltype(result)> == Size);
   bool ok = true;
   for (size_t i = 0; i < Size; ++i)
      ok = Close(GetValue(result, i), expected[i], label) && ok;
   return ok;
}

bool CheckNumericCurlStorage()
{
   static_assert(StaticMatrix<curl_numeric_test::StaticMatrix>);
   // Component-major gradients have deliberately unequal off-diagonal entries.
   const std::array<std::array<float, 3>, 3> matrix_float{
      {{1, 2.5f, -4}, {3, 5, 6.75f}, {7.25f, 8.5f, 10}}};
   bool ok = CheckNumericCurl(matrix_float, std::array<float, 3>{1.75f, -11.25f, .5f},
      "nested-array float vector gradient curl");
   static_assert(StaticMatrix<curl_numeric_test::ProxyMatrix>);
   // Curl must materialize scalar entries before subtracting the readable proxies.
   ok = CheckNumericCurl(curl_numeric_test::ProxyMatrix{matrix_float},
      std::array<float, 3>{1.75f, -11.25f, .5f}, "proxy float vector gradient curl") && ok;
   SerialRecursiveArray<float, 3, 3> recursive{};
   for (size_t i = 0; i < 3; ++i)
   for (size_t j = 0; j < 3; ++j) { recursive(i, j) = matrix_float[i][j]; }
   ok = CheckNumericCurl(recursive, std::array<float, 3>{1.75f, -11.25f, .5f},
      "recursive float vector gradient curl") && ok;
   const std::array<std::array<long double, 3>, 3> matrix3{
      {{1, 2.5L, -4}, {3, 5, 6.75L}, {7.25L, 8.5L, 10}}};
   ok = CheckNumericCurl(matrix3, std::array<long double, 3>{1.75L, -11.25L, .5L},
      "nested-array 3D vector gradient curl") && ok;
   const curl_numeric_test::StaticMatrix custom_matrix{
      {{{-2, 3.25L, 5.5L}, {7.75L, 11, -13}, {17, 19.5L, 23}}}};
   return CheckNumericCurl(custom_matrix, std::array<long double, 3>{32.5L, -11.5L, 4.5L},
      "custom static-matrix vector gradient curl") && ok;
}

struct NativeScalar
{
   Real value;
   operator Real() const { return value; }
};
NativeScalar operator-(NativeScalar&& value) { return {-value.value}; }
NativeScalar operator-(const NativeScalar&) = delete;

bool CheckNegation()
{
   const auto native = MakeCoefficient<"native">([] { return NativeScalar{2}; });
   const auto native_result = (-native)(Empty{}, Empty{}, Empty{}, Empty{}, Empty{}, Empty{});
   bool ok = Close(native_result.value, -2, "native rvalue unary minus");
   const auto matrix = MakeMatrixCoefficient<"matrix">(
      [] { return Matrix<2>{{{1, -2}, {3.5, 0}}}; });
   const auto array_result = (-matrix)(Empty{}, Empty{}, Empty{}, Empty{}, Empty{}, Empty{});
   const Matrix<2> expected{{{-1, 2}, {-3.5, 0}}};
   for (size_t c = 0; c < 2; ++c)
   for (size_t d = 0; d < 2; ++d)
      ok = Close(array_result[c][d], expected[c][d], "nested-array unary minus") && ok;
   return ok;
}

struct MultiplicationProxy { Real value; };
struct AssignableElement
{
   Real value{};
   AssignableElement& operator=(MultiplicationProxy result)
   { value = result.value; return *this; }
};
struct ScalarFactor { Real value; };
MultiplicationProxy operator*(const ScalarFactor& scalar, const AssignableElement& element)
{ return {scalar.value*element.value}; }
[[maybe_unused]] MultiplicationProxy operator*(const AssignableElement& element, const ScalarFactor& scalar)
{ return {element.value*scalar.value}; }

bool CheckAssignableProductProxy()
{
   SerialRecursiveArray<AssignableElement, 2> elements{};
   elements(0).value = 2; elements(1).value = 3;
   const auto left = ScalarFactor{4}*elements;
   const auto right = elements*ScalarFactor{4};
   bool ok = Close(left(0).value, 8, "assignable product proxy");
   ok = Close(left(1).value, 12, "assignable product proxy") && ok;
   ok = Close(right(0).value, 8, "right assignable product proxy") && ok;
   return Close(right(1).value, 12, "right assignable product proxy") && ok;
}

template<class Value, size_t Components>
bool CheckValue(const Value& value, const Coordinates<Components>& expected,
                const char* label)
{
   bool ok = true;
   for (size_t c = 0; c < Components; ++c)
      ok = Close(GetValue(value, c), expected[c], label) && ok;
   return ok;
}

template<class Channel, class Pullback, class Point, size_t Components, size_t Dim, class Fields = Empty>
bool CheckPhysical(const Pullback& pullback, const Point& point,
                   const Gradient<Components, Dim>& seed, Real scale, const char* label,
                   const Fields& fields = {})
{
   const auto value = pullback.template Get<Channel>()(
      TestKernel{}, Empty{}, Empty{}, Empty{}, point, fields);
   bool ok = true;
   for (size_t c = 0; c < Components; ++c)
   for (size_t d = 0; d < Dim; ++d)
      ok = Close(GetValue(value, c, d), scale*seed[c][d], label) && ok;
   return ok;
}

template<class Row, size_t Components, size_t Dim>
bool CheckMapped(const Row& row, const Gradient<Components, Dim>& seed,
                 Real scale, const Matrix<Dim>& inverse, const char* label)
{
   Gradient<Components, Dim> actual{};
   [&]<size_t... I>(std::index_sequence<I...>)
      { ((actual[I] = std::get<I>(row.gradients).values), ...); }
         (std::make_index_sequence<Components>{});
   bool ok = true;
   for (size_t c = 0; c < Components; ++c)
   for (size_t d = 0; d < Dim; ++d)
   {
      Real expected = 0;
      for (size_t physical = 0; physical < Dim; ++physical)
         expected += scale*seed[c][physical]*inverse[physical][d];
      ok = Close(actual[c][d], expected, label) && ok;
   }
   return ok;
}

template<class Pullback, class Point, size_t Components, size_t Dim, class Fields = Empty>
bool CheckCurrent(const Pullback& pullback, const Point& point,
                  const Gradient<Components, Dim>& seed, Real scale,
                  const Matrix<Dim>& inverse, const char* label, const Fields& fields = {})
{
   static_assert(Pullback::template Contains<GradientChannel>());
   static_assert(!Pullback::template Contains<ValueChannel>());
   auto row = Field(Gradient<Components, Dim>{});
   ComputeContributions(TestKernel{}, Empty{}, Empty{}, Empty{}, point, fields,
                        pullback, TensorIndex<Dim>{}, row);
   return CheckPhysical<GradientChannel>(pullback, point, seed, scale, label, fields) &&
          CheckMapped(row, seed, scale, inverse, label);
}

template<class Pullback, class Point, size_t Components, size_t Dim>
bool CheckGlobal(const Pullback& pullback, const Point& point,
                 const Gradient<Components, Dim>& seed, Real minus_scale, Real plus_scale,
                 const Matrix<Dim>& minus_inverse, const Matrix<Dim>& plus_inverse,
                 const char* label)
{
   static_assert(Pullback::template Contains<GradientMinusChannel>());
   static_assert(Pullback::template Contains<GradientPlusChannel>());
   auto minus_row = Field(Gradient<Components, Dim>{});
   auto plus_row = Field(Gradient<Components, Dim>{});
   ComputeContributions(TestKernel{}, Empty{}, Empty{}, Empty{}, point, Empty{},
                        pullback, TensorIndex<Dim>{}, minus_row, plus_row);
   bool ok = CheckPhysical<GradientMinusChannel>(pullback, point, seed, minus_scale, label);
   ok = CheckPhysical<GradientPlusChannel>(pullback, point, seed, plus_scale, label) && ok;
   ok = CheckMapped(minus_row, seed, minus_scale, minus_inverse, label) && ok;
   return CheckMapped(plus_row, seed, plus_scale, plus_inverse, label) && ok;
}

template<class Test, class Coefficient, size_t Components, size_t Dim>
bool CheckPullbacks(const Test& test, const Coefficient& coefficient,
                   const Gradient<Components, Dim>& adjoint,
                   const Matrix<Dim>& minus_inverse, const Matrix<Dim>& plus_inverse)
{
   const auto curl_test = curl(test);
   static_assert(std::is_same_v<std::remove_cvref_t<decltype(curl_test)>,
                               CurlExpr<Test>>);
   auto pair = [&](const auto& expression) { return dot(coefficient, expression); };
   QuadraturePointContext<TensorIndex<Dim>, Coordinates<Dim>, Matrix<Dim>> cell{};
   cell.inv_J_mesh = minus_inverse; cell.weight = 2; cell.det_J = 3;
   BoundaryFacetQuadraturePointContext<TensorIndex<Dim>, Coordinates<Dim>, Matrix<Dim>, Coordinates<Dim>> boundary{};
   boundary.inv_J_mesh = minus_inverse; boundary.weight = 2; boundary.det_J_facet = 5;
   TwoSidedFacetQuadraturePointContext<TensorIndex<Dim>, Coordinates<Dim>, Matrix<Dim>, Matrix<Dim>, Coordinates<Dim>> interior{};
   interior.inv_J_mesh_minus = minus_inverse; interior.inv_J_mesh_plus = plus_inverse;
   interior.weight = 2; interior.det_J_facet = 5;

   const auto cell_pullback = MakeCellPullback(integrate(Cells<"mesh">{}, pair(curl_test)));
   bool ok = CheckCurrent(cell_pullback, cell, adjoint, 6, minus_inverse, "cell curl");
   {
      const auto physical_seed = cell_pullback.template Get<GradientChannel>()(
         TestKernel{}, Empty{}, Empty{}, Empty{}, cell, Empty{});
      // For G={{1,2,3},{4,5,6},{7,8,10}}, curl(G)=(2,-4,2).
      // The coefficient (2,-3,5) gives 26. This independent adjoint
      // identity catches reversing every sign in the skew matrix.
      const Matrix<3> G{{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}};
      Real contraction = 0;
      for (size_t c = 0; c < 3; ++c)
      for (size_t d = 0; d < 3; ++d)
         contraction += GetValue(physical_seed, c, d)*G[c][d];
      ok = Close(contraction, 6*26, "3D curl adjoint identity") && ok;
   }
   ok = CheckCurrent(MakeBoundaryFacetPullback(integrate(BoundaryFacets<"mesh">{}, pair(curl_test))),
      boundary, adjoint, 10, minus_inverse, "boundary curl") && ok;
   ok = CheckCurrent(MakeBoundaryFacetPullback(integrate(BoundaryFacets<"mesh">{}, pair(average(curl_test)))),
      boundary, adjoint, 10, minus_inverse, "boundary average curl") && ok;
   ok = CheckCurrent(MakeLocalInteriorFacetPullback(integrate(InteriorFacets<"mesh">{}, pair(average(curl_test)))),
      interior, adjoint, 5, minus_inverse, "local average curl") && ok;
   ok = CheckCurrent(MakeLocalInteriorFacetPullback(integrate(InteriorFacets<"mesh">{}, pair(jump(curl_test)))),
      interior, adjoint, 10, minus_inverse, "local jump curl") && ok;
   ok = CheckGlobal(MakeGlobalInteriorFacetPullback(integrate(InteriorFacets<"mesh">{}, pair(average(curl_test)))),
      interior, adjoint, 5, 5, minus_inverse, plus_inverse, "global average curl") && ok;
   ok = CheckGlobal(MakeGlobalInteriorFacetPullback(integrate(InteriorFacets<"mesh">{}, pair(jump(curl_test)))),
      interior, adjoint, 10, -10, minus_inverse, plus_inverse, "global jump curl") && ok;
   const auto explicit_sides = MakeGlobalInteriorFacetPullback(integrate(InteriorFacets<"mesh">{},
      2*pair(minus(curl_test)) + 3*pair(plus(curl_test))));
   return CheckGlobal(explicit_sides, interior, adjoint, 20, 30,
      minus_inverse, plus_inverse, "explicit curl sides") && ok;
}

template<class Minus, class Plus>
struct FacetFields { Minus minus_fields; Plus plus_fields; };

template<class Trial, size_t Components, size_t Dim, size_t CurlComponents>
bool CheckEvaluation(const Trial& trial, const Matrix<Dim>& minus_inverse, const Matrix<Dim>& plus_inverse,
                     const Gradient<Components, Dim>& trial_minus, const Gradient<Components, Dim>& trial_plus,
                     const Gradient<Components, Dim>& bound_minus, const Gradient<Components, Dim>& bound_plus,
                     const Coordinates<CurlComponents>& expected_trial_minus,
                     const Coordinates<CurlComponents>& expected_trial_plus,
                     const Coordinates<CurlComponents>& expected_bound_minus,
                     const Coordinates<CurlComponents>& expected_bound_plus)
{
   const auto minus_fields = MakeStaticMap(MakeEntry<NameTag<"u">>(Field(trial_minus)),
                                          MakeEntry<NameTag<"w">>(Field(bound_minus)));
   const auto plus_fields = MakeStaticMap(MakeEntry<NameTag<"u">>(Field(trial_plus)),
                                         MakeEntry<NameTag<"w">>(Field(bound_plus)));
   const FacetFields fields{minus_fields, plus_fields};
   TwoSidedFacetQuadraturePointContext<TensorIndex<Dim>, Coordinates<Dim>, Matrix<Dim>, Matrix<Dim>, Coordinates<Dim>> interior{};
   interior.inv_J_mesh_minus = minus_inverse; interior.inv_J_mesh_plus = plus_inverse;
   const FiniteElementField<"w"> bound;
   auto evaluate = [&](const auto& expression, const auto& point, const auto& values)
   { return expression(TestKernel{}, Empty{}, Empty{}, Empty{}, point, values); };
   bool ok = CheckValue(evaluate(curl(trial), interior.MinusSide(), minus_fields), expected_trial_minus, "trial curl");
   ok = CheckValue(evaluate(curl(bound), interior.PlusSide(), plus_fields), expected_bound_plus, "bound curl") && ok;
   const auto trial_curl = curl(trial);
   const auto bound_curl = curl(bound);
   ok = CheckValue(evaluate(minus(trial_curl), interior, fields), expected_trial_minus, "minus trial curl") && ok;
   ok = CheckValue(evaluate(plus(trial_curl), interior, fields), expected_trial_plus, "plus trial curl") && ok;
   ok = CheckValue(evaluate(minus(bound_curl), interior, fields), expected_bound_minus, "minus bound curl") && ok;
   ok = CheckValue(evaluate(plus(bound_curl), interior, fields), expected_bound_plus, "plus bound curl") && ok;
   Coordinates<CurlComponents> trial_mean{}, bound_jump{};
   for (size_t c = 0; c < CurlComponents; ++c)
   {
      trial_mean[c] = .5*(expected_trial_minus[c]+expected_trial_plus[c]);
      bound_jump[c] = expected_bound_minus[c]-expected_bound_plus[c];
   }
   ok = CheckValue(evaluate(average(trial_curl), interior, fields), trial_mean, "average trial curl") && ok;
   ok = CheckValue(evaluate(jump(bound_curl), interior, fields), bound_jump, "jump bound curl") && ok;

   const TestSpace<"v", field_shape_v<Trial>> test;
   auto form = [&](const auto& field)
   {
      return integrate(Cells<"mesh">{}, dot(curl(field), curl(test)));
   };
   const auto trial_pullback = MakeCellPullback(form(trial));
   const auto bound_pullback = MakeCellPullback(form(bound));
   static_assert(ContainsOriginalExpression<decltype(trial_pullback.template Get<GradientChannel>()),
      CurlExpr<Trial>>());
   static_assert(ContainsOriginalExpression<decltype(bound_pullback.template Get<GradientChannel>()),
      CurlExpr<FiniteElementField<"w">>>());
   auto adjoint = [](const Coordinates<CurlComponents>& seed)
   {
      Gradient<Components, Dim> result{};
      result[0][1] = -seed[2]; result[0][2] = seed[1];
      result[1][0] = seed[2]; result[1][2] = -seed[0];
      result[2][0] = -seed[1]; result[2][1] = seed[0];
      return result;
   };
   QuadraturePointContext<TensorIndex<Dim>, Coordinates<Dim>, Matrix<Dim>> cell{};
   cell.inv_J_mesh = minus_inverse; cell.weight = 2; cell.det_J = 3;
   ok = CheckCurrent(trial_pullback, cell, adjoint(expected_trial_minus), 6, minus_inverse,
      "original trial curl seed", minus_fields) && ok;
   return CheckCurrent(bound_pullback, cell, adjoint(expected_bound_minus), 6, minus_inverse,
      "original bound curl seed", minus_fields) && ok;
}

bool RunChecks()
{
   const auto seed3 = MakeVectorCoefficient<"seed3">([] { return Coordinates<3>{2, -3, 5}; });
   const Matrix<3> minus3{{{2, 1, -1}, {3, -2, 4}, {1, 5, 2}}};
   const Matrix<3> plus3{{{1, -3, 2}, {4, 2, -1}, {-2, 1, 3}}};
   bool ok = CheckPullbacks(VectorTestSpace<"v">{}, seed3,
      Gradient<3, 3>{{{0, -5, -3}, {5, 0, -2}, {3, 2, 0}}}, minus3, plus3);
   return CheckEvaluation(VectorTrialSpace<"u">{}, minus3, plus3,
      Gradient<3, 3>{{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}},
      Gradient<3, 3>{{{-1, 2, 1}, {3, -2, 4}, {2, 5, -3}}},
      Gradient<3, 3>{{{-1, -2, -3}, {-4, -5, -6}, {-7, -8, -10}}},
      Gradient<3, 3>{{{2, -4, -2}, {-6, 4, -8}, {-4, -10, 6}}},
      Coordinates<3>{4, 5, -4}, Coordinates<3>{17, 26, 18},
      Coordinates<3>{-4, -5, 4}, Coordinates<3>{-34, -52, -36}) && ok;
}
} // namespace

int main()
{
   const bool curl_ok = RunChecks();
   const bool products_ok = CheckAssignableProductProxy();
   const bool storage_ok = CheckNumericCurlStorage();
   return CheckNegation() && curl_ok && products_ok && storage_ok ? 0 : 1;
}
