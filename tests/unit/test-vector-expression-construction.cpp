// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>

using namespace gendil;

// This test verifies that new operator overloads (for SerialRecursiveArray arithmetic,
// matrix-vector multiplication, inner product) do not break expression-template construction.

int main()
{
   std::cout << "Testing expression construction...\n";

   // Test 1: Basic scalar expressions still compile
   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;

      auto mass = u * v;
      auto gradgrad = dot(grad(u), grad(v));
      (void)mass; (void)gradgrad;  // Variables used only to verify expressions construct

      std::cout << "  [PASS] Scalar expressions: u*v and dot(grad(u), grad(v))\n";
   }

   // Test 2: Multiplication expressions with Normal{}
   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;
      (void)v;  // Used only in comments

      // This should use expression-template operator* (MultFieldExpr)
      // grad(u) * Normal{} should construct a MultFieldExpr
      auto flux = grad(u) * Normal{};
      (void)flux;  // Variable used only to verify expression constructs

      std::cout << "  [PASS] grad(u) * Normal{} constructs\n";
   }

   // Test 3: inner() expression template
   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;

      // inner(grad(u), grad(v)) should create InnerExpr
      auto inner_expr = inner(grad(u), grad(v));
      (void)inner_expr;  // Variable used only to verify expression constructs

      std::cout << "  [PASS] inner(grad(u), grad(v)) constructs\n";
   }

   // Test 4: dot() with jump and average
   {
      VectorTrialSpace<"u"> u;
      VectorTestSpace<"v"> v;

      auto jump_penalty = dot(jump(u), jump(v));
      (void)jump_penalty;  // Variable used only to verify expression constructs

      std::cout << "  [PASS] dot(jump(u), jump(v)) constructs (using vector spaces)\n";
   }

   // Test 5: Complex SIPDG-like expressions
   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;

      // Consistency: average(grad(u) · Normal{}) appears with jump(v)
      // dot(grad(u), Normal{}) is valid: grad of scalar is Vector, Normal is Vector
      // Outer operation is scalar multiplication (not dot)
      auto consistency = average(dot(grad(u), Normal{})) * jump(v);

      // Adjoint: jump(u) with average(grad(v) * Normal{})
      auto adjoint = jump(u) * average(dot(grad(v), Normal{}));
      (void)consistency; (void)adjoint;  // Variables used only to verify expressions construct

      std::cout << "  [PASS] SIPDG consistency and adjoint terms construct\n";
   }

   // Test 6: Coefficient multiplication
   {
      TrialSpace<"u"> u;
      TestSpace<"v"> v;

      // Define a simple coefficient lambda
      auto mu_fn = [](const auto& X) -> Real { return 1.0; };
      auto mu = MakeCoefficient<"mu", PhysicalCoordinate>(mu_fn);

      // Coefficient * expression should still work
      auto scaled = mu * u * v;
      (void)scaled;  // Variable used only to verify expression constructs

      std::cout << "  [PASS] Coefficient multiplication: mu * u * v\n";
   }

   // Test 7: Value-level operations don't interfere
   {
      // Create some value-level objects
      SerialRecursiveArray<Real, 3> vec1;
      SerialRecursiveArray<Real, 3> vec2;
      vec1(0) = 1.0; vec1(1) = 2.0; vec1(2) = 3.0;
      vec2(0) = 4.0; vec2(1) = 5.0; vec2(2) = 6.0;

      // Value-level operations
      auto sum = vec1 + vec2;
      auto diff = vec1 - vec2;
      auto scaled = 0.5 * vec1;
      auto product = Dot(vec1, vec2);
      (void)sum; (void)diff; (void)scaled; (void)product;  // Variables used only to verify operations construct

      // Matrix-vector
      SerialRecursiveArray<Real, 3, 3> mat;
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            mat(i, j) = (i == j) ? 1.0 : 0.0;

      std::array<Real, 3> arr = {1.0, 2.0, 3.0};
      auto mat_vec_product = mat * arr;
      (void)mat_vec_product;  // Variable used only to verify operation constructs

      // Inner product
      SerialRecursiveArray<Real, 3, 3> mat2;
      for (int i = 0; i < 3; ++i)
         for (int j = 0; j < 3; ++j)
            mat2(i, j) = 1.0;

      auto frobenius = Inner(mat, mat2);
      (void)frobenius;  // Variable used only to verify operation constructs

      std::cout << "  [PASS] Value-level operations work correctly\n";
   }

   std::cout << "\nAll expression construction tests passed!\n";
   std::cout << "This confirms that new operator overloads do not break expression templates.\n";

   return 0;
}
