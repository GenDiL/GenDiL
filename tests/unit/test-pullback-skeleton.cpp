// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/FiniteElementMethod/WeakForm/pullback.hpp>
#include <gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writechannelcontributions.hpp>

#include <cmath>
#include <iostream>

using namespace gendil;

template<class T>
struct is_product_expr : std::false_type {};

template<class L, class R>
struct is_product_expr<ProductExpr<L, R>> : std::true_type {};

template<class T>
inline constexpr bool is_product_expr_v =
   is_product_expr<std::remove_cvref_t<T>>::value;

// This test verifies the pullback skeleton implementation:
// - Channel-set deduction with shape-compatible seeds
// - Compile-time channel presence queries
// - Channel access via get<>()
// - Seed shape validation

int main()
{
   std::cout << "Testing pullback skeleton...\n";

   // Test spaces
   TrialSpace<"u"> u;                                   // Scalar trial space
   TestSpace<"v"> v;                                    // Scalar test space
   TestSpace<"vv", FieldShape::Vector> vv;             // Vector test space

   // Shape-compatible seeds
   auto c = MakeCoefficient<"c", PhysicalCoordinate>(
      [](const auto&) { return Real{1.0}; });

   auto mu = MakeCoefficient<"mu", PhysicalCoordinate>(
      [](const auto&) { return Real{2.0}; });

   auto beta = MakeVectorCoefficient<"beta", PhysicalCoordinate>(
      [](const auto&) {
         return std::array<Real, 3>{Real{1.0}, Real{0.0}, Real{0.0}};
      });

   auto A = MakeMatrixCoefficient<"A", PhysicalCoordinate>(
      [](const auto&) {
         std::array<std::array<Real, 3>, 3> result = {};
         result[0][0] = Real{1.0};
         result[1][1] = Real{1.0};
         result[2][2] = Real{1.0};
         return result;
      });

   // Test 1: TestSpace (Scalar) + scalar seed → ValueChannel
   {
      auto result_v = pullback(v, c);

      // Channel presence checks
      static_assert(decltype(result_v)::contains<ValueChannel>(),
         "pullback(v, c) should have ValueChannel");
      static_assert(!decltype(result_v)::contains<GradientChannel>(),
         "pullback(v, c) should NOT have GradientChannel");

      // Channel access check
      [[maybe_unused]] auto& value_seed = result_v.template get<ValueChannel>();

      std::cout << "  [PASS] pullback(TestSpace<Scalar>, scalar_seed) → ValueChannel\n";
   }

   // Test 2: grad(TestSpace<Scalar>) (Vector) + vector seed → GradientChannel
   {
      auto result_grad_v = pullback(grad(v), beta);

      // Channel presence checks
      static_assert(decltype(result_grad_v)::contains<GradientChannel>(),
         "pullback(grad(v), beta) should have GradientChannel");
      static_assert(!decltype(result_grad_v)::contains<ValueChannel>(),
         "pullback(grad(v), beta) should NOT have ValueChannel");

      // Channel access check
      [[maybe_unused]] auto& gradient_seed = result_grad_v.template get<GradientChannel>();

      std::cout << "  [PASS] pullback(grad(TestSpace<Scalar>), vector_seed) → GradientChannel\n";
   }

   // Test 3: TestSpace<Vector> (Vector) + vector seed → ValueChannel
   {
      auto result_vv = pullback(vv, beta);

      // Channel presence checks
      static_assert(decltype(result_vv)::contains<ValueChannel>(),
         "pullback(vv, beta) should have ValueChannel");
      static_assert(!decltype(result_vv)::contains<GradientChannel>(),
         "pullback(vv, beta) should NOT have GradientChannel");

      // Channel access check
      [[maybe_unused]] auto& value_seed = result_vv.template get<ValueChannel>();

      std::cout << "  [PASS] pullback(TestSpace<Vector>, vector_seed) → ValueChannel\n";
   }

   // Test 4: grad(TestSpace<Vector>) (Matrix) + matrix seed → GradientChannel
   {
      auto result_grad_vv = pullback(grad(vv), A);

      // Channel presence checks
      static_assert(decltype(result_grad_vv)::contains<GradientChannel>(),
         "pullback(grad(vv), A) should have GradientChannel");
      static_assert(!decltype(result_grad_vv)::contains<ValueChannel>(),
         "pullback(grad(vv), A) should NOT have ValueChannel");

      // Channel access check
      [[maybe_unused]] auto& gradient_seed = result_grad_vv.template get<GradientChannel>();

      std::cout << "  [PASS] pullback(grad(TestSpace<Vector>), matrix_seed) → GradientChannel\n";
   }

   // Test 4b: vector normal-flux ProductExpr + vector seed → GradientChannel
   {
      auto normal_flux = grad(vv) * Normal{};
      static_assert(is_product_expr_v<decltype(normal_flux)>);
      static_assert(decltype(normal_flux)::product_kind == ProductKind::MatVec);

      auto result_flux = pullback(normal_flux, beta);

      static_assert(decltype(result_flux)::contains<GradientChannel>(),
         "pullback(grad(VectorTestSpace) * Normal{}, vector_seed) should have GradientChannel");
      static_assert(!decltype(result_flux)::contains<ValueChannel>(),
         "pullback(grad(VectorTestSpace) * Normal{}, vector_seed) should NOT have ValueChannel");

      [[maybe_unused]] auto& gradient_seed =
         result_flux.template get<GradientChannel>();
      using GradientSeed = std::remove_cvref_t<decltype(gradient_seed)>;
      static_assert(field_shape_v<GradientSeed> == FieldShape::Matrix,
         "MatVec ProductExpr pullback should seed vector test gradients with a matrix-shaped expression.");
      static_assert(is_test_free_v<GradientSeed>,
         "MatVec ProductExpr pullback gradient seed should be test-free.");

      std::cout << "  [PASS] pullback(grad(TestSpace<Vector>) * Normal{}, vector_seed) → GradientChannel\n";
   }

   // Test 5: dot(beta, vv) → ValueChannel
   {
      auto expr = dot(beta, vv);
      auto result = pullback(expr, c);

      // Channel presence checks
      static_assert(decltype(result)::contains<ValueChannel>(),
         "pullback(dot(beta, vv), c) should have ValueChannel");
      static_assert(!decltype(result)::contains<GradientChannel>(),
         "pullback(dot(beta, vv), c) should NOT have GradientChannel");

      // Channel access
      [[maybe_unused]] auto& value = result.template get<ValueChannel>();

      // REQUIRED: Verify channel coefficient is test-free and correctly shaped
      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Vector,
         "ValueChannel coefficient should be Vector-shaped");
      static_assert(is_test_free_v<ValueType>,
         "ValueChannel coefficient should be test-free");

      std::cout << "  [PASS] pullback(dot(beta, vv), c) → ValueChannel (test-free Vector)\n";
   }

   // Test 6: dot(vv, beta) → ValueChannel (symmetric)
   {
      auto expr = dot(vv, beta);
      auto result = pullback(expr, c);

      static_assert(decltype(result)::contains<ValueChannel>());
      static_assert(!decltype(result)::contains<GradientChannel>());

      std::cout << "  [PASS] pullback(dot(vv, beta), c) → ValueChannel\n";
   }

   // Test 7: dot(grad(u), grad(v)) → GradientChannel
   {
      auto expr = dot(grad(u), grad(v));
      auto result = pullback(expr, c);

      // Channel presence checks
      static_assert(decltype(result)::contains<GradientChannel>(),
         "pullback(dot(grad(u), grad(v)), c) should have GradientChannel");
      static_assert(!decltype(result)::contains<ValueChannel>(),
         "pullback(dot(grad(u), grad(v)), c) should NOT have ValueChannel");

      // Channel access
      [[maybe_unused]] auto& grad_value = result.template get<GradientChannel>();

      // REQUIRED: Verify channel coefficient is test-free and correctly shaped
      using GradType = std::remove_cvref_t<decltype(grad_value)>;
      static_assert(field_shape_v<GradType> == FieldShape::Vector,
         "GradientChannel coefficient should be Vector-shaped");
      static_assert(is_test_free_v<GradType>,
         "GradientChannel coefficient should be test-free");

      std::cout << "  [PASS] pullback(dot(grad(u), grad(v)), c) → GradientChannel (test-free Vector)\n";
   }

   // Test 8: dot(grad(v), grad(u)) → GradientChannel (symmetric)
   {
      auto expr = dot(grad(v), grad(u));
      auto result = pullback(expr, c);

      static_assert(decltype(result)::contains<GradientChannel>());
      static_assert(!decltype(result)::contains<ValueChannel>());

      std::cout << "  [PASS] pullback(dot(grad(v), grad(u)), c) → GradientChannel\n";
   }

   // Invalid cases (commented, documented)
   // Should fail: no test-linear operand
   // auto invalid_free = pullback(dot(beta, beta), c);

   // Should fail: nonlinear in test
   // auto invalid_nonlinear = pullback(dot(vv, vv), c);
   // auto invalid_grad_nonlinear = pullback(dot(grad(v), grad(v)), c);

   // Should fail: seed shape mismatch (DotExpr is Scalar, seed must be Scalar)
   // auto invalid_seed = pullback(dot(beta, vv), beta);

   // ========================================================================
   // ProductExpr ScalarTimes tests
   // ========================================================================

   // Test 9: u * v → ValueChannel (scalar mass, test-free on left)
   {
      auto expr = u * v;
      auto result = pullback(expr, c);

      static_assert(decltype(result)::contains<ValueChannel>());
      static_assert(!decltype(result)::contains<GradientChannel>());

      [[maybe_unused]] auto& value = result.template get<ValueChannel>();

      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Scalar,
         "ValueChannel coefficient should be Scalar");
      static_assert(is_test_free_v<ValueType>,
         "ValueChannel coefficient should be test-free");

      std::cout << "  [PASS] pullback(u * v, c) → ValueChannel (test-free Scalar)\n";
   }

   // Test 10: v * u → ValueChannel (scalar mass, test-free on right)
   {
      auto expr = v * u;
      auto result = pullback(expr, c);

      static_assert(decltype(result)::contains<ValueChannel>());
      static_assert(!decltype(result)::contains<GradientChannel>());

      [[maybe_unused]] auto& value = result.template get<ValueChannel>();

      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Scalar);
      static_assert(is_test_free_v<ValueType>);

      std::cout << "  [PASS] pullback(v * u, c) → ValueChannel (test-free Scalar)\n";
   }

   // Test 11: mu * grad(v) → GradientChannel
   {
      auto expr = mu * grad(v);
      auto result = pullback(expr, beta);

      static_assert(decltype(result)::contains<GradientChannel>());
      static_assert(!decltype(result)::contains<ValueChannel>());

      [[maybe_unused]] auto& grad_value = result.template get<GradientChannel>();

      using GradType = std::remove_cvref_t<decltype(grad_value)>;
      static_assert(field_shape_v<GradType> == FieldShape::Vector,
         "GradientChannel coefficient should be Vector");
      static_assert(is_test_free_v<GradType>,
         "GradientChannel coefficient should be test-free");

      std::cout << "  [PASS] pullback(mu * grad(v), beta) → GradientChannel (test-free Vector)\n";
   }

   // Test 12: grad(v) * mu → GradientChannel (symmetric)
   {
      auto expr = grad(v) * mu;
      auto result = pullback(expr, beta);

      static_assert(decltype(result)::contains<GradientChannel>());
      static_assert(!decltype(result)::contains<ValueChannel>());

      [[maybe_unused]] auto& grad_value = result.template get<GradientChannel>();

      using GradType = std::remove_cvref_t<decltype(grad_value)>;
      static_assert(field_shape_v<GradType> == FieldShape::Vector);
      static_assert(is_test_free_v<GradType>);

      std::cout << "  [PASS] pullback(grad(v) * mu, beta) → GradientChannel (test-free Vector)\n";
   }

   // Test 13: mu * vv → ValueChannel (scalar coeff times vector test)
   {
      auto expr = mu * vv;
      auto result = pullback(expr, beta);

      static_assert(decltype(result)::contains<ValueChannel>());
      static_assert(!decltype(result)::contains<GradientChannel>());

      [[maybe_unused]] auto& value = result.template get<ValueChannel>();

      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Vector,
         "ValueChannel coefficient should be Vector");
      static_assert(is_test_free_v<ValueType>,
         "ValueChannel coefficient should be test-free");

      std::cout << "  [PASS] pullback(mu * vv, beta) → ValueChannel (test-free Vector)\n";
   }

   // Test 14: vv * mu → ValueChannel (symmetric)
   {
      auto expr = vv * mu;
      auto result = pullback(expr, beta);

      static_assert(decltype(result)::contains<ValueChannel>());
      static_assert(!decltype(result)::contains<GradientChannel>());

      [[maybe_unused]] auto& value = result.template get<ValueChannel>();

      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Vector);
      static_assert(is_test_free_v<ValueType>);

      std::cout << "  [PASS] pullback(vv * mu, beta) → ValueChannel (test-free Vector)\n";
   }

   // Test 15: mu * grad(vv) → GradientChannel
   {
      auto expr = mu * grad(vv);
      auto result = pullback(expr, A);

      static_assert(decltype(result)::contains<GradientChannel>());
      static_assert(!decltype(result)::contains<ValueChannel>());

      [[maybe_unused]] auto& grad_value = result.template get<GradientChannel>();

      using GradType = std::remove_cvref_t<decltype(grad_value)>;
      static_assert(field_shape_v<GradType> == FieldShape::Matrix,
         "GradientChannel coefficient should be Matrix");
      static_assert(is_test_free_v<GradType>,
         "GradientChannel coefficient should be test-free");

      std::cout << "  [PASS] pullback(mu * grad(vv), A) → GradientChannel (test-free Matrix)\n";
   }

   // Test 16: grad(vv) * mu → GradientChannel (symmetric)
   {
      auto expr = grad(vv) * mu;
      auto result = pullback(expr, A);

      static_assert(decltype(result)::contains<GradientChannel>());
      static_assert(!decltype(result)::contains<ValueChannel>());

      [[maybe_unused]] auto& grad_value = result.template get<GradientChannel>();

      using GradType = std::remove_cvref_t<decltype(grad_value)>;
      static_assert(field_shape_v<GradType> == FieldShape::Matrix);
      static_assert(is_test_free_v<GradType>);

      std::cout << "  [PASS] pullback(grad(vv) * mu, A) → GradientChannel (test-free Matrix)\n";
   }

   // Test 16b: dot(A * grad(v), n) uses A^T n as the grad(v) seed
   {
      auto nonsymmetric_A = MakeMatrixCoefficient<"nonsymmetric_A">(
         [] ()
         {
            std::array<std::array<Real, 2>, 2> result{};
            result[0][0] = Real{2.0};
            result[0][1] = -Real{0.35};
            result[1][0] = Real{0.65};
            result[1][1] = Real{1.4};
            return result;
         });

      auto expr = dot(nonsymmetric_A * grad(v), Normal{});
      auto result = pullback(expr, ScaleExpr{1.0});

      static_assert(decltype(result)::contains<GradientChannel>());
      static_assert(!decltype(result)::contains<ValueChannel>());

      const auto& grad_seed = result.template get<GradientChannel>();

      struct FakeFacetQuadPointContext
      {
         std::array<Real, 2> physical_normal;
      };

      const FakeFacetQuadPointContext qctx{
         std::array<Real, 2>{Real{0.6}, -Real{0.8}}};
      const Empty empty{};

      const auto seed_q =
         grad_seed(empty, empty, empty, empty, qctx, empty);

      const std::array<Real, 2> expected_AT_n{
         Real{2.0} * qctx.physical_normal[0] +
            Real{0.65} * qctx.physical_normal[1],
        -Real{0.35} * qctx.physical_normal[0] +
            Real{1.4} * qctx.physical_normal[1]};

      const std::array<Real, 2> wrong_A_n{
         Real{2.0} * qctx.physical_normal[0] -
            Real{0.35} * qctx.physical_normal[1],
         Real{0.65} * qctx.physical_normal[0] +
            Real{1.4} * qctx.physical_normal[1]};

      const Real err =
         std::abs(seed_q(0) - expected_AT_n[0]) +
         std::abs(seed_q(1) - expected_AT_n[1]);
      const Real wrong_sep =
         std::abs(seed_q(0) - wrong_A_n[0]) +
         std::abs(seed_q(1) - wrong_A_n[1]);

      if (err > Real{1e-14} || wrong_sep < Real{1e-3})
      {
         std::cerr
            << "FAILED: MatVec pullback seed should be A^T n, not A n.\n"
            << "  seed = [" << seed_q(0) << ", " << seed_q(1) << "]\n"
            << "  A^T n = [" << expected_AT_n[0] << ", "
            << expected_AT_n[1] << "]\n"
            << "  A n = [" << wrong_A_n[0] << ", "
            << wrong_A_n[1] << "]\n";
         return 1;
      }

      using GradSeedType = std::remove_cvref_t<decltype(grad_seed)>;
      static_assert(field_shape_v<GradSeedType> == FieldShape::Vector);
      static_assert(is_test_free_v<GradSeedType>);

      std::cout
         << "  [PASS] pullback(dot(A * grad(v), n), 1) → "
         << "GradientChannel with A^T n seed\n";
   }

   // Invalid ProductExpr cases (commented, documented)
   // Should fail: no test-linear operand
   // auto invalid_free = pullback(mu * u, c);

   // Should fail: nonlinear in test
   // auto invalid_nonlinear = pullback(v * v, c);

   // Should fail: nonlinear MatVec ProductExpr with two test-linear operands
   // auto invalid_matvec = pullback(grad(vv) * vv, beta);

   // Should fail: seed shape mismatch
   // auto invalid_seed = pullback(u * v, beta);

   // ========================================================================
   // SumExpr pullback tests
   // ========================================================================

   // Test 17: Mixed value + gradient channels
   {
      auto expr = (u * v) + dot(grad(u), grad(v));
      auto result = pullback(expr, c);

      // Channel presence checks
      static_assert(decltype(result)::contains<ValueChannel>(),
         "pullback(u*v + dot(grad(u), grad(v)), c) should have ValueChannel");
      static_assert(decltype(result)::contains<GradientChannel>(),
         "pullback(u*v + dot(grad(u), grad(v)), c) should have GradientChannel");

      // Channel access
      [[maybe_unused]] auto& value = result.template get<ValueChannel>();
      [[maybe_unused]] auto& grad_value = result.template get<GradientChannel>();

      // Verify channel coefficient shapes and test-freedom
      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Scalar,
         "ValueChannel coefficient should be Scalar");
      static_assert(is_test_free_v<ValueType>,
         "ValueChannel coefficient should be test-free");

      using GradType = std::remove_cvref_t<decltype(grad_value)>;
      static_assert(field_shape_v<GradType> == FieldShape::Vector,
         "GradientChannel coefficient should be Vector");
      static_assert(is_test_free_v<GradType>,
         "GradientChannel coefficient should be test-free");

      std::cout << "  [PASS] pullback(u*v + dot(grad(u), grad(v)), c) → ValueChannel + GradientChannel\n";
   }

   // Test 18: Same-channel value merge
   {
      auto expr = (u * v) + (mu * v);
      auto result = pullback(expr, c);

      // Channel presence checks
      static_assert(decltype(result)::contains<ValueChannel>(),
         "pullback(u*v + mu*v, c) should have ValueChannel");
      static_assert(!decltype(result)::contains<GradientChannel>(),
         "pullback(u*v + mu*v, c) should NOT have GradientChannel");

      // Channel access
      [[maybe_unused]] auto& value = result.template get<ValueChannel>();

      // Verify channel coefficient is a sum expression (merged)
      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Scalar,
         "ValueChannel coefficient should be Scalar");
      static_assert(is_test_free_v<ValueType>,
         "ValueChannel coefficient should be test-free");
      static_assert(is_sum_expr_v<ValueType>,
         "ValueChannel coefficient should be SumExpr (merged from two terms)");

      std::cout << "  [PASS] pullback(u*v + mu*v, c) → ValueChannel (merged SumExpr)\n";
   }

   // Test 19: Same-channel gradient merge
   {
      auto expr = dot(grad(u), grad(v)) + dot(beta, grad(v));
      auto result = pullback(expr, c);

      // Channel presence checks
      static_assert(decltype(result)::contains<GradientChannel>(),
         "pullback(dot(grad(u), grad(v)) + dot(beta, grad(v)), c) should have GradientChannel");
      static_assert(!decltype(result)::contains<ValueChannel>(),
         "pullback(dot(grad(u), grad(v)) + dot(beta, grad(v)), c) should NOT have ValueChannel");

      // Channel access
      [[maybe_unused]] auto& grad_value = result.template get<GradientChannel>();

      // Verify channel coefficient
      using GradType = std::remove_cvref_t<decltype(grad_value)>;
      static_assert(field_shape_v<GradType> == FieldShape::Vector,
         "GradientChannel coefficient should be Vector");
      static_assert(is_test_free_v<GradType>,
         "GradientChannel coefficient should be test-free");
      static_assert(is_sum_expr_v<GradType>,
         "GradientChannel coefficient should be SumExpr (merged from two terms)");

      std::cout << "  [PASS] pullback(dot(grad(u), grad(v)) + dot(beta, grad(v)), c) → GradientChannel (merged SumExpr)\n";
   }

   // Test 20: Flattened three-term sum
   {
      auto expr = (u * v) + (mu * v) + dot(grad(u), grad(v));
      auto result = pullback(expr, c);

      // Channel presence checks
      static_assert(decltype(result)::contains<ValueChannel>(),
         "pullback(three-term sum, c) should have ValueChannel");
      static_assert(decltype(result)::contains<GradientChannel>(),
         "pullback(three-term sum, c) should have GradientChannel");

      // Channel access
      [[maybe_unused]] auto& value = result.template get<ValueChannel>();
      [[maybe_unused]] auto& grad_value = result.template get<GradientChannel>();

      // Verify channel coefficients
      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Scalar,
         "ValueChannel coefficient should be Scalar");
      static_assert(is_test_free_v<ValueType>,
         "ValueChannel coefficient should be test-free");

      using GradType = std::remove_cvref_t<decltype(grad_value)>;
      static_assert(field_shape_v<GradType> == FieldShape::Vector,
         "GradientChannel coefficient should be Vector");
      static_assert(is_test_free_v<GradType>,
         "GradientChannel coefficient should be test-free");

      std::cout << "  [PASS] pullback(u*v + mu*v + dot(grad(u), grad(v)), c) → both channels\n";
   }

   // Test 21: Vector value channel merge
   {
      // Create another scalar coefficient for this test
      auto nu = MakeCoefficient<"nu", PhysicalCoordinate>(
         [](const auto&) { return Real{3.0}; });

      auto expr = (mu * vv) + (nu * vv);
      auto result = pullback(expr, beta);  // Use vector seed since expr is Vector-shaped

      // Channel presence checks
      static_assert(decltype(result)::contains<ValueChannel>(),
         "pullback(mu*vv + nu*vv, beta) should have ValueChannel");
      static_assert(!decltype(result)::contains<GradientChannel>(),
         "pullback(mu*vv + nu*vv, beta) should NOT have GradientChannel");

      // Channel access
      [[maybe_unused]] auto& value = result.template get<ValueChannel>();

      // Verify channel coefficient
      using ValueType = std::remove_cvref_t<decltype(value)>;
      static_assert(field_shape_v<ValueType> == FieldShape::Vector,
         "ValueChannel coefficient should be Vector");
      static_assert(is_test_free_v<ValueType>,
         "ValueChannel coefficient should be test-free");
      static_assert(is_sum_expr_v<ValueType>,
         "ValueChannel coefficient should be SumExpr (merged from two terms)");

      std::cout << "  [PASS] pullback(mu*vv + nu*vv, beta) → ValueChannel (merged Vector SumExpr)\n";
   }

   // Invalid SumExpr cases (commented, documented)
   // Should fail: unsupported term InnerExpr pullback not implemented
   // auto invalid_inner = pullback(inner(grad(u), grad(v)) + (u * v), c);

   // Should fail: seed shape mismatch for MatVec ProductExpr
   // auto invalid_matvec_seed = pullback(A * vv, c);

   // Should fail: seed shape mismatch
   // auto invalid_seed = pullback((u * v) + dot(grad(u), grad(v)), beta);

   std::cout << "\nAll pullback skeleton tests passed!\n";
   std::cout << "Verified:\n";
   std::cout << "  - Primitive pullback rules with shape-compatible seeds\n";
   std::cout << "  - DotExpr pullback with bilinear adjoint rule\n";
   std::cout << "  - ProductExpr ScalarTimes and MatVec pullback adjoints\n";
   std::cout << "  - SumExpr pullback with channel merge\n";

   return 0;
}
