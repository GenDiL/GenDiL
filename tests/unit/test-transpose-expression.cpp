// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

using Matrix23 = std::array<std::array<Real, 3>, 2>;
using Vector2 = std::array<Real, 2>;
using Vector3 = std::array<Real, 3>;

Matrix23 TestMatrix()
{
   return Matrix23{{
      {{Real{2.0}, -Real{0.35}, Real{1.10}}},
      {{Real{0.65}, Real{1.40}, -Real{0.20}}}
   }};
}

Vector2 TestVector2()
{
   return Vector2{Real{0.60}, -Real{0.80}};
}

Vector3 TestVector3()
{
   return Vector3{Real{1.25}, -Real{0.50}, Real{0.75}};
}

template<typename Value>
bool Close(const Value actual, const Value expected, const Real tol = 1e-14)
{
   return std::abs(actual - expected) <= tol;
}

} // namespace

int main()
{
   std::cout << "Testing public transpose expression...\n";

   const Empty empty{};

   // Scalar transpose: identity shape and value.
   {
      auto scalar = ScaleExpr{Real{2.75}};
      auto scalar_t = transpose(scalar);

      static_assert(field_shape_v<decltype(scalar_t)> == FieldShape::Scalar);
      static_assert(is_test_free_v<decltype(scalar_t)>);

      const Real value = scalar_t(empty, empty, empty, empty, empty, empty);
      if (!Close(value, Real{2.75}))
      {
         std::cerr << "FAILED: transpose(scalar) changed the value.\n";
         return 1;
      }

      std::cout << "  [PASS] transpose(scalar) is value-preserving\n";
   }

   // Scalar transpose propagates named-field requirements and side-dependency.
   {
      auto scalar_coeff =
         MakeCoefficient<"scalar_depends_on_a", FieldValue<"a">>(
            [] (const Real a) { return Real{1.0} + a; });
      auto scalar_t = transpose(scalar_coeff);

      using Reqs = coefficient_named_field_requirements_t<decltype(scalar_t)>;
      static_assert(contains_named_field_requirement_v<Reqs, "a">);
      using AReq = find_named_field_requirement_t<Reqs, "a">;
      static_assert(need_values(AReq::mask));
      static_assert(
         has_provenance(
            AReq::provenance,
            NamedFieldProvenance::CoefficientInput));
      static_assert(has_unqualified_side_dependent_inputs_v<decltype(scalar_t)>);
      static_assert(is_test_free_v<decltype(scalar_t)>);

      std::cout
         << "  [PASS] transpose(scalar coefficient) propagates named-field traits\n";
   }

   // Matrix transpose: matrix<M,N> -> matrix<N,M>.
   {
      auto matrix =
         MakeMatrixCoefficient<"A23">([] () { return TestMatrix(); });
      auto vector =
         MakeVectorCoefficient<"x2">([] () { return TestVector2(); });

      auto matrix_t = transpose(matrix);
      auto product = matrix_t * vector;

      static_assert(field_shape_v<decltype(matrix_t)> == FieldShape::Matrix);
      static_assert(field_shape_v<decltype(product)> == FieldShape::Vector);

      const auto transposed =
         matrix_t(empty, empty, empty, empty, empty, empty);
      static_assert(
         static_num_rows_v<decltype(transposed)> == 3 &&
         static_num_cols_v<decltype(transposed)> == 2);

      const Matrix23 A = TestMatrix();
      for (Integer i = 0; i < 2; ++i)
      {
         for (Integer j = 0; j < 3; ++j)
         {
            if (!Close(transposed(j, i), A[i][j]))
            {
               std::cerr << "FAILED: transpose(matrix) entry mismatch.\n";
               return 1;
            }
         }
      }

      const auto y = product(empty, empty, empty, empty, empty, empty);
      const Vector2 x = TestVector2();
      const Vector3 expected{
         A[0][0] * x[0] + A[1][0] * x[1],
         A[0][1] * x[0] + A[1][1] * x[1],
         A[0][2] * x[0] + A[1][2] * x[1]};

      for (Integer i = 0; i < 3; ++i)
      {
         if (!Close(y(i), expected[i]))
         {
            std::cerr << "FAILED: transpose(A) * x did not evaluate as A^T x.\n";
            return 1;
         }
      }

      auto matrix_tt = transpose(matrix_t);
      const auto round_trip =
         matrix_tt(empty, empty, empty, empty, empty, empty);
      static_assert(
         static_num_rows_v<decltype(round_trip)> == 2 &&
         static_num_cols_v<decltype(round_trip)> == 3);

      for (Integer i = 0; i < 2; ++i)
      {
         for (Integer j = 0; j < 3; ++j)
         {
            if (!Close(round_trip(i, j), A[i][j]))
            {
               std::cerr
                  << "FAILED: transpose(transpose(A)) did not evaluate as A.\n";
               return 1;
            }
         }
      }

      std::cout << "  [PASS] transpose(matrix) evaluates and composes correctly\n";
   }

   // Vector transpose: vector<N> -> matrix<1,N>.
   {
      auto vector =
         MakeVectorCoefficient<"v3">([] () { return TestVector3(); });
      auto other =
         MakeVectorCoefficient<"w3">(
            [] ()
            {
               return Vector3{Real{-0.20}, Real{1.10}, Real{0.40}};
            });

      auto row = transpose(vector);
      auto row_times_vector = row * other;

      static_assert(field_shape_v<decltype(row)> == FieldShape::Matrix);
      static_assert(
         field_shape_v<decltype(row_times_vector)> == FieldShape::Vector);

      const auto row_q = row(empty, empty, empty, empty, empty, empty);
      static_assert(
         static_num_rows_v<decltype(row_q)> == 1 &&
         static_num_cols_v<decltype(row_q)> == 3);

      const Vector3 v = TestVector3();
      for (Integer i = 0; i < 3; ++i)
      {
         if (!Close(row_q(0, i), v[i]))
         {
            std::cerr << "FAILED: transpose(vector) entry mismatch.\n";
            return 1;
         }
      }

      const auto product_q =
         row_times_vector(empty, empty, empty, empty, empty, empty);
      const Real expected =
         v[0] * Real{-0.20} + v[1] * Real{1.10} + v[2] * Real{0.40};
      if (!Close(product_q(0), expected))
      {
         std::cerr
            << "FAILED: transpose(vector) * vector did not produce row-vector product.\n";
         return 1;
      }

      std::cout << "  [PASS] transpose(vector) evaluates as a 1xN row matrix\n";
   }

   // Matrix coefficient input requirements propagate through transpose.
   {
      auto matrix_depends_on_a =
         MakeMatrixCoefficient<"A_of_a", FieldValue<"a">>(
            [] (const Real a)
            {
               std::array<std::array<Real, 2>, 2> A{};
               A[0][0] = Real{1.0} + a;
               A[0][1] = Real{0.25};
               A[1][0] = -Real{0.50};
               A[1][1] = Real{2.0} - Real{0.25} * a;
               return A;
            });

      using Reqs =
         coefficient_named_field_requirements_t<
            decltype(transpose(matrix_depends_on_a))>;
      static_assert(contains_named_field_requirement_v<Reqs, "a">);
      using AReq = find_named_field_requirement_t<Reqs, "a">;
      static_assert(need_values(AReq::mask));
      static_assert(
         has_provenance(
            AReq::provenance,
            NamedFieldProvenance::CoefficientInput));

      std::cout
         << "  [PASS] transpose(matrix coefficient) propagates named-field traits\n";
   }

   // Test-linearity and plus-side Jacobian requirements propagate transparently.
   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;

      static_assert(is_test_linear_v<decltype(transpose(grad(v)))>);
      static_assert(
         requires_plus_side_jacobian_v<decltype(transpose(average(grad(u))))>);

      std::cout
         << "  [PASS] transpose propagates test-linearity and plus-Jacobian traits\n";
   }

   std::cout << "All public transpose expression tests passed.\n";
   return 0;
}
