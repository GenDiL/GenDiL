// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <gendil/FiniteElementMethod/WeakForm/pullback.hpp>
#include <gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperator.hpp>

#include <iostream>
#include <cmath>

using namespace gendil;

// Runtime validation for pullback-based GenericCellIntegrandOperatorPullback
//
// Validates:
//   1. u * v (value-only) - compare pullback vs old path
//   2. dot(grad(u), grad(v)) (gradient-only) - compare pullback vs old path
//   3. u*v + dot(grad(u), grad(v)) (mixed) - compare pullback vs split reference

// Error utilities
template <typename VectorType>
Real AbsoluteL2Error(const VectorType& a, const VectorType& b)
{
   GENDIL_VERIFY(a.Size() == b.Size(), "Vector sizes do not match.");
   Real err_sq = 0.0;
   for (Integer i = 0; i < a.Size(); ++i)
   {
      const Real d = a[i] - b[i];
      err_sq += d * d;
   }
   return std::sqrt(err_sq);
}

template <typename VectorType>
Real L2Norm(const VectorType& x)
{
   Real norm_sq = 0.0;
   for (Integer i = 0; i < x.Size(); ++i)
   {
      norm_sq += x[i] * x[i];
   }
   return std::sqrt(norm_sq);
}

template <typename VectorType>
Real RelativeL2Error(const VectorType& a, const VectorType& b)
{
   const Real abs_err = AbsoluteL2Error(a, b);
   const Real norm_b = L2Norm(b);
   if (norm_b == 0.0)
   {
      return abs_err;
   }
   return abs_err / norm_b;
}

int main()
{
   std::cout << "Runtime validation: pullback vs old GenericOperator paths\n";

   // Setup: 1D mesh, order 1, scalar test space
   constexpr Integer Dim = 1;
   constexpr Integer Order = 1;

   const Integer n = 3;  // 3 elements
   const Real h = 1.0 / Real(n);
   CartesianMesh<Dim> mesh({n}, {h}, {0.0});

   FiniteElementOrders<Order> orders;
   auto fe = MakeLobattoFiniteElement(orders);
   auto fes = MakeFiniteElementSpace(mesh, fe);

   const Integer ndofs = fes.GetNumberOfFiniteElementDofs();

   // Integration rule
   constexpr Integer num_quad_1d = Order + 3;
   IntegrationRuleNumPoints<num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   // Input DoFs (simple non-constant function for testing)
   Vector dofs_in_vec(ndofs);
   {
      Real* data = dofs_in_vec.WriteHostData();
      for (Integer i = 0; i < ndofs; ++i) {
         data[i] = 1.0 + Real(i) * 0.5;
      }
   }

   // Test spaces and domain
   TrialSpace<"u"> u;
   TestSpace<"v"> v;
   Cells<"mesh1"> cells;

   // Weak form context (trial field + test field + domain)
   // Note: Both trial and test fields must be registered explicitly since they have different names
   auto wf_ctx = MakeWeakFormContext(
      MakeTrialField<"u">(fes),
      MakeTestField<"v">(fes),
      MakeDomain<"mesh1">(mesh)
   );

   using KernelPolicy = SerialKernelConfiguration;

   bool all_pass = true;
   constexpr Real tol = 1e-12;
   constexpr Real nonzero_tol = 1e-14;  // Tolerance for nonzero output check

   // Debug: print input vector
   std::cout << "\nInput DoF vector:";
   {
      const Real* in_data = dofs_in_vec.ReadHostData();
      for (Integer i = 0; i < ndofs; ++i) {
         std::cout << " " << in_data[i];
      }
      std::cout << "\n";
   }

   // Test 1: Value-only integrand (u * v)
   {
      std::cout << "\n=== Test 1: Value-only integrand (u * v) ===\n";

      auto integrand = integrate(cells, u * v);

      // Old path
      auto old_op = MakeGenericOperator<KernelPolicy>(integrand, wf_ctx, integration_rule);
      Vector output_old_vec(ndofs);
      output_old_vec = 0.0;
      old_op(dofs_in_vec, output_old_vec);

      // Pullback path
      auto pb_op = MakePullbackGenericOperator<KernelPolicy>(integrand, wf_ctx, integration_rule);
      Vector output_pullback_vec(ndofs);
      output_pullback_vec = 0.0;
      pb_op(dofs_in_vec, output_pullback_vec);

      // Compare and check for non-trivial outputs
      const Real norm_old = L2Norm(output_old_vec);
      const Real norm_pullback = L2Norm(output_pullback_vec);
      const Real abs_err = AbsoluteL2Error(output_old_vec, output_pullback_vec);
      const Real rel_err = RelativeL2Error(output_pullback_vec, output_old_vec);

      std::cout << "  Norm (old path):     " << norm_old << "\n";
      std::cout << "  Norm (pullback):     " << norm_pullback << "\n";
      std::cout << "  Max abs error:       " << abs_err << "\n";
      std::cout << "  Relative error:      " << rel_err << "\n";

      // Check for non-trivial output
      if (norm_old <= nonzero_tol) {
         std::cout << "  [FAIL] Old path output is effectively zero (norm = " << norm_old << ")\n";
         all_pass = false;
      } else if (norm_pullback <= nonzero_tol) {
         std::cout << "  [FAIL] Pullback output is effectively zero (norm = " << norm_pullback << ")\n";
         all_pass = false;
      } else if (abs_err > tol) {
         std::cout << "  [FAIL] Pullback differs from old path\n";
         all_pass = false;
      } else {
         std::cout << "  [PASS] Pullback matches old path\n";
      }
   }

   // Test 2: Gradient-only integrand (dot(grad(u), grad(v)))
   {
      std::cout << "\n=== Test 2: Gradient-only integrand (dot(grad(u), grad(v))) ===\n";

      auto integrand = integrate(cells, dot(grad(u), grad(v)));

      // Old path
      auto old_op = MakeGenericOperator<KernelPolicy>(integrand, wf_ctx, integration_rule);
      Vector output_old_vec(ndofs);
      output_old_vec = 0.0;
      old_op(dofs_in_vec, output_old_vec);

      // Pullback path
      auto pb_op = MakePullbackGenericOperator<KernelPolicy>(integrand, wf_ctx, integration_rule);
      Vector output_pullback_vec(ndofs);
      output_pullback_vec = 0.0;
      pb_op(dofs_in_vec, output_pullback_vec);

      // Compare and check for non-trivial outputs
      const Real norm_old = L2Norm(output_old_vec);
      const Real norm_pullback = L2Norm(output_pullback_vec);
      const Real abs_err = AbsoluteL2Error(output_old_vec, output_pullback_vec);
      const Real rel_err = RelativeL2Error(output_pullback_vec, output_old_vec);

      std::cout << "  Norm (old path):     " << norm_old << "\n";
      std::cout << "  Norm (pullback):     " << norm_pullback << "\n";
      std::cout << "  Max abs error:       " << abs_err << "\n";
      std::cout << "  Relative error:      " << rel_err << "\n";

      // Check for non-trivial output
      if (norm_old <= nonzero_tol) {
         std::cout << "  [FAIL] Old path output is effectively zero (norm = " << norm_old << ")\n";
         all_pass = false;
      } else if (norm_pullback <= nonzero_tol) {
         std::cout << "  [FAIL] Pullback output is effectively zero (norm = " << norm_pullback << ")\n";
         all_pass = false;
      } else if (abs_err > tol) {
         std::cout << "  [FAIL] Pullback differs from old path\n";
         all_pass = false;
      } else {
         std::cout << "  [PASS] Pullback matches old path\n";
      }
   }

   // Test 3: Mixed integrand (u*v + dot(grad(u), grad(v)))
   {
      std::cout << "\n=== Test 3: Mixed integrand (u*v + dot(grad(u), grad(v))) ===\n";

      auto integrand_mixed = integrate(cells, u * v + dot(grad(u), grad(v)));
      auto integrand_val = integrate(cells, u * v);
      auto integrand_grad = integrate(cells, dot(grad(u), grad(v)));

      // Reference: split into two old-path operators
      auto old_val_op = MakeGenericOperator<KernelPolicy>(integrand_val, wf_ctx, integration_rule);
      auto old_grad_op = MakeGenericOperator<KernelPolicy>(integrand_grad, wf_ctx, integration_rule);

      Vector output_ref_val_vec(ndofs);
      Vector output_ref_grad_vec(ndofs);
      output_ref_val_vec = 0.0;
      output_ref_grad_vec = 0.0;

      old_val_op(dofs_in_vec, output_ref_val_vec);
      old_grad_op(dofs_in_vec, output_ref_grad_vec);

      // Combine reference outputs
      Vector output_ref_vec(ndofs);
      {
         Real* ref_data = output_ref_vec.WriteHostData();
         const Real* val_data = output_ref_val_vec.ReadHostData();
         const Real* grad_data = output_ref_grad_vec.ReadHostData();
         for (Integer i = 0; i < ndofs; ++i) {
            ref_data[i] = val_data[i] + grad_data[i];
         }
      }

      // Pullback path: single call with mixed integrand
      auto pb_mixed_op = MakePullbackGenericOperator<KernelPolicy>(integrand_mixed, wf_ctx, integration_rule);
      Vector output_pullback_vec(ndofs);
      output_pullback_vec = 0.0;
      pb_mixed_op(dofs_in_vec, output_pullback_vec);

      // Compare with additional norm checks for reference components
      const Real norm_ref_val = L2Norm(output_ref_val_vec);
      const Real norm_ref_grad = L2Norm(output_ref_grad_vec);
      const Real norm_ref = L2Norm(output_ref_vec);
      const Real norm_pullback = L2Norm(output_pullback_vec);
      const Real abs_err = AbsoluteL2Error(output_ref_vec, output_pullback_vec);
      const Real rel_err = RelativeL2Error(output_pullback_vec, output_ref_vec);

      std::cout << "  Norm (ref value):    " << norm_ref_val << "\n";
      std::cout << "  Norm (ref grad):     " << norm_ref_grad << "\n";
      std::cout << "  Norm (ref mixed):    " << norm_ref << "\n";
      std::cout << "  Norm (pullback):     " << norm_pullback << "\n";
      std::cout << "  Max abs error:       " << abs_err << "\n";
      std::cout << "  Relative error:      " << rel_err << "\n";

      // Check for non-trivial outputs
      if (norm_ref_val <= nonzero_tol) {
         std::cout << "  [FAIL] Reference value output is effectively zero (norm = " << norm_ref_val << ")\n";
         all_pass = false;
      } else if (norm_ref_grad <= nonzero_tol) {
         std::cout << "  [FAIL] Reference gradient output is effectively zero (norm = " << norm_ref_grad << ")\n";
         all_pass = false;
      } else if (norm_ref <= nonzero_tol) {
         std::cout << "  [FAIL] Reference mixed output is effectively zero (norm = " << norm_ref << ")\n";
         all_pass = false;
      } else if (norm_pullback <= nonzero_tol) {
         std::cout << "  [FAIL] Pullback output is effectively zero (norm = " << norm_pullback << ")\n";
         all_pass = false;
      } else if (abs_err > tol) {
         std::cout << "  [FAIL] Pullback differs from split reference\n";
         all_pass = false;
      } else {
         std::cout << "  [PASS] Pullback matches split reference (NEW capability)\n";
         std::cout << "  [INFO] Mixed value+gradient integrand now works\n";
      }
   }

   if (all_pass) {
      std::cout << "\nAll runtime validation tests PASSED!\n";
      std::cout << "Verified:\n";
      std::cout << "  - Value-only pullback numerically matches old path\n";
      std::cout << "  - Gradient-only pullback numerically matches old path\n";
      std::cout << "  - Mixed value+gradient pullback matches split reference (NEW)\n";
      return 0;
   } else {
      std::cout << "\nSome runtime validation tests FAILED\n";
      return 1;
   }
}
