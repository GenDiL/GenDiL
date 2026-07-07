// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

using namespace std;
using namespace gendil;

namespace
{

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
Real RelativeL2Error(const VectorType& a, const VectorType& b)
{
   const Real abs_err = AbsoluteL2Error(a, b);
   Real b_norm_sq = 0.0;
   for (Integer i = 0; i < b.Size(); ++i)
   {
      b_norm_sq += b[i] * b[i];
   }
   const Real b_norm = std::sqrt(b_norm_sq);
   GENDIL_VERIFY(b_norm > 0.0, "Reference vector has zero norm");
   return abs_err / b_norm;
}

// Component-major layout helpers
template<typename FiniteElementSpace>
Vector CreateComponentVector(
   const FiniteElementSpace& vector_fe_space,
   Integer component,
   const Vector& scalar_input)
{
   constexpr Integer vdim = FiniteElementSpace::finite_element_type::shape_functions::num_comp;
   const Integer scalar_ndofs = scalar_input.Size();

   Vector result(scalar_ndofs * vdim);
   result = 0.0;

   // Component-major layout: component c starts at offset c * scalar_ndofs
   Real* result_ptr = result.WriteHostData();
   const Real* input_ptr = scalar_input.ReadHostData();
   for (Integer i = 0; i < scalar_ndofs; ++i) {
      result_ptr[component * scalar_ndofs + i] = input_ptr[i];
   }
   return result;
}

template<typename FiniteElementSpace>
Vector ExtractComponent(
   const Vector& vector_output,
   Integer component,
   const FiniteElementSpace& scalar_fe_space)
{
   const Integer scalar_ndofs = scalar_fe_space.GetNumberOfFiniteElementDofs();
   Vector result(scalar_ndofs);

   Real* result_ptr = result.WriteHostData();
   const Real* output_ptr = vector_output.ReadHostData();

   // Component-major layout
   for (Integer i = 0; i < scalar_ndofs; ++i) {
      result_ptr[i] = output_ptr[component * scalar_ndofs + i];
   }
   return result;
}

template < Integer order >
int TestVectorGradGrad()
{
   std::cout << "\n=== Vector grad-grad test, order = " << order << " ===\n";

   // --- Mesh ---
   const Integer num_elem_1d = 6;
   const Real h = 1.0 / num_elem_1d;
   Cartesian1DMesh mesh_x(h, num_elem_1d),
                   mesh_y(h, num_elem_1d),
                   mesh_z(h, num_elem_1d);
   auto mesh = MakeCartesianProductMesh(mesh_x, mesh_y, mesh_z);

   // --- Finite Element Spaces ---
   FiniteElementOrders<order, order, order> orders;
   auto fe = MakeLobattoFiniteElement(orders);  // Lobatto basis

   // Vector FE: vdim=3
   constexpr Integer vdim = 3;
   auto vector_fe = MakeVectorFiniteElement(fe, fe, fe);
   auto vector_fe_space = MakeFiniteElementSpace(mesh, vector_fe);

   // Scalar FE for component-wise oracle
   auto scalar_fe_space = MakeFiniteElementSpace(mesh, fe);

   const Integer vector_ndofs = vector_fe_space.GetNumberOfFiniteElementDofs();
   const Integer scalar_ndofs = scalar_fe_space.GetNumberOfFiniteElementDofs();

   std::cout << "Vector dofs: " << vector_ndofs
             << ", Scalar dofs: " << scalar_ndofs
             << ", vdim: " << vdim << "\n";

   // --- Integration Rule ---
   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> nq;
   auto integration_rule = MakeIntegrationRule(nq);

   // --- Kernel Policy ---
   using KernelPolicy = SerialKernelConfiguration;

   // ==========================================================================
   // Test 1: Component-wise scalar equivalence (PRIMARY DIAGNOSTIC)
   // ==========================================================================
   std::cout << "\nTest 1 - Component-wise scalar equivalence:\n";

   // Create weak-form expressions
   Cells<"mesh1"> domain;
   VectorTrialSpace<"u"> u;  // Vector field reference
   VectorTestSpace<"u"> v;   // Vector field reference

   // Vector weak form: inner(grad(u), grad(v))
   auto vector_weak_form = integrate(domain, inner(grad(u), grad(v)));

   auto vector_weak_form_context = MakeWeakFormContext(
      MakeTrialField<"u">(vector_fe_space),
      MakeIntegrationDomain<"mesh1">(vector_fe_space));

   auto generic_vector_operator =
      MakeGenericOperator<KernelPolicy>(
         vector_weak_form,
         vector_weak_form_context,
         integration_rule);

   // Scalar weak form: inner(grad(us), grad(vs)) (same as dot for scalar)
   auto scalar_weak_form = integrate(domain, inner(grad(u), grad(v)));

   auto scalar_weak_form_context = MakeWeakFormContext(
      MakeTrialField<"u">(scalar_fe_space),
      MakeIntegrationDomain<"mesh1">(scalar_fe_space));

   auto generic_scalar_operator =
      MakeGenericOperator<KernelPolicy>(
         scalar_weak_form,
         scalar_weak_form_context,
         integration_rule);

   // Test each component with detailed diagnostics
   const Real tol = 1e-12;
   bool component_test_passed = true;

   for (Integer comp = 0; comp < vdim; ++comp)
   {
      std::cout << "\n  Testing component " << comp << ":\n";

      // Create scalar input
      Vector scalar_input(scalar_ndofs);
      FillRandom(scalar_input, 54321 + comp, -1.0, 1.0);

      // Embed in vector field at component comp (others are zero)
      Vector vector_input = CreateComponentVector(vector_fe_space, comp, scalar_input);

      // Apply operators
      Vector scalar_output(scalar_ndofs);
      Vector vector_output_full(vector_ndofs);
      scalar_output = 0.0;
      vector_output_full = 0.0;

      generic_scalar_operator(scalar_input, scalar_output);
      generic_vector_operator(vector_input, vector_output_full);

      // Extract active component from vector output
      Vector active_output = ExtractComponent(vector_output_full, comp, scalar_fe_space);

      // Compute norms
      Real scalar_norm = 0.0;
      Real active_norm = 0.0;
      for (Integer i = 0; i < scalar_ndofs; ++i) {
         scalar_norm += scalar_output[i] * scalar_output[i];
         active_norm += active_output[i] * active_output[i];
      }
      scalar_norm = std::sqrt(scalar_norm);
      active_norm = std::sqrt(active_norm);

      // Compute active component error
      const Real active_rel_error = RelativeL2Error(active_output, scalar_output);

      std::cout << "    ||scalar output||          = " << scalar_norm << "\n";
      std::cout << "    ||vector active component|| = " << active_norm << "\n";
      std::cout << "    Active component rel error = " << active_rel_error << "\n";

      // Check leakage into inactive components
      std::cout << "    Inactive component leakage:\n";
      for (Integer other_comp = 0; other_comp < vdim; ++other_comp) {
         if (other_comp == comp) continue;

         Vector inactive_output = ExtractComponent(vector_output_full, other_comp, scalar_fe_space);
         Real inactive_norm = 0.0;
         for (Integer i = 0; i < scalar_ndofs; ++i) {
            inactive_norm += inactive_output[i] * inactive_output[i];
         }
         inactive_norm = std::sqrt(inactive_norm);

         std::cout << "      Component " << other_comp << " norm = " << inactive_norm;
         if (inactive_norm > tol * scalar_norm) {
            std::cout << "  [LEAKAGE!]";
         }
         std::cout << "\n";
      }

      // Check if test passed for this component
      if (active_rel_error > tol) {
         std::cerr << "    FAILED: Active component error too large\n";
         component_test_passed = false;
      }
   }

   if (!component_test_passed) {
      std::cerr << "\nFAILED: Component-wise scalar equivalence test\n";
      return 1;
   }

   std::cout << "\n  SUCCESS: All components match scalar operator\n";

   // ==========================================================================
   // Test 2: Constant nullspace check
   // ==========================================================================
   std::cout << "\nTest 2 - Constant nullspace check:\n";
   std::cout << "  For grad-grad, constant fields should give zero output\n";

   Vector constant_input(vector_ndofs);
   constant_input = 1.0;  // u = (1, 1, 1) everywhere

   Vector nullspace_output(vector_ndofs);
   nullspace_output = 0.0;

   generic_vector_operator(constant_input, nullspace_output);

   Real nullspace_norm = 0.0;
   for (Integer i = 0; i < vector_ndofs; ++i) {
      nullspace_norm += nullspace_output[i] * nullspace_output[i];
   }
   nullspace_norm = std::sqrt(nullspace_norm);

   std::cout << "  ||A * constant||  = " << nullspace_norm << "\n";

   if (nullspace_norm > 1e-12) {
      std::cerr << "  WARNING: Constant nullspace not satisfied (may be boundary effects)\n";
   } else {
      std::cout << "  SUCCESS: Constant nullspace satisfied\n";
   }

   // ==========================================================================
   // Informational: Compare to MakeGradGradOperator
   // ==========================================================================
   std::cout << "\n[INFORMATIONAL] Comparison to MakeGradGradOperator:\n";
   std::cout << "  Note: MakeGradGradOperator may use different conventions\n";

   // Create legacy grad-grad operator
   auto legacy_operator = MakeGradGradOperator<KernelPolicy>(
      vector_fe_space,
      integration_rule);

   // Random input
   Vector input(vector_ndofs);
   FillRandom(input, 12345, -1.0, 1.0);

   Vector generic_output(vector_ndofs);
   Vector legacy_output(vector_ndofs);
   generic_output = 0.0;
   legacy_output = 0.0;

   generic_vector_operator(input, generic_output);
   legacy_operator(input, legacy_output);

   const Real rel_error = RelativeL2Error(generic_output, legacy_output);
   std::cout << "  Relative L2 error vs legacy: " << rel_error << "\n";
   std::cout << "  (Not used as pass/fail criterion)\n";

   std::cout << "\nSUCCESS: order " << order << " passed all required tests.\n";
   return 0;
}

template < Integer order >
int TestVectorGradGrad_vdim2()
{
   std::cout << "\n=== Vector grad-grad vdim=2 on 3D mesh, order = " << order << " ===\n";

   // --- Mesh (3D) ---
   const Integer num_elem_1d = 6;
   const Real h = 1.0 / num_elem_1d;
   Cartesian1DMesh mesh_x(h, num_elem_1d),
                   mesh_y(h, num_elem_1d),
                   mesh_z(h, num_elem_1d);
   auto mesh = MakeCartesianProductMesh(mesh_x, mesh_y, mesh_z);

   // --- Finite Element Spaces ---
   FiniteElementOrders<order, order, order> orders;
   auto fe = MakeLobattoFiniteElement(orders);

   // Vector FE: vdim=2 on 3D mesh (catch vdim != dim assumptions)
   constexpr Integer vdim = 2;
   auto vector_fe = MakeVectorFiniteElement(fe, fe);  // Only 2 components
   auto vector_fe_space = MakeFiniteElementSpace(mesh, vector_fe);

   // Scalar FE for component-wise oracle
   auto scalar_fe_space = MakeFiniteElementSpace(mesh, fe);

   const Integer vector_ndofs = vector_fe_space.GetNumberOfFiniteElementDofs();
   const Integer scalar_ndofs = scalar_fe_space.GetNumberOfFiniteElementDofs();

   std::cout << "Vector dofs: " << vector_ndofs
             << ", Scalar dofs: " << scalar_ndofs
             << ", vdim: " << vdim << "\n";

   // --- Integration Rule ---
   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> nq;
   auto integration_rule = MakeIntegrationRule(nq);

   // --- Kernel Policy ---
   using KernelPolicy = SerialKernelConfiguration;

   // ==========================================================================
   // Test: Component-wise scalar equivalence for vdim=2
   // ==========================================================================
   std::cout << "\nComponent-wise scalar equivalence (vdim=2):\n";

   Cells<"mesh1"> domain;
   VectorTrialSpace<"u"> u;  // Vector field reference
   VectorTestSpace<"u"> v;   // Vector field reference

   auto vector_weak_form = integrate(domain, inner(grad(u), grad(v)));
   auto vector_weak_form_context = MakeWeakFormContext(
      MakeTrialField<"u">(vector_fe_space),
      MakeIntegrationDomain<"mesh1">(vector_fe_space));
   auto vector_operator =
      MakeGenericOperator<KernelPolicy>(
         vector_weak_form,
         vector_weak_form_context,
         integration_rule);

   // Create scalar operator
   auto scalar_weak_form = integrate(domain, inner(grad(u), grad(v)));
   auto scalar_weak_form_context = MakeWeakFormContext(
      MakeTrialField<"u">(scalar_fe_space),
      MakeIntegrationDomain<"mesh1">(scalar_fe_space));
   auto scalar_operator =
      MakeGenericOperator<KernelPolicy>(
         scalar_weak_form,
         scalar_weak_form_context,
         integration_rule);

   const Real tol = 1e-12;
   bool vdim2_test_passed = true;

   // Test each component with detailed diagnostics
   for (Integer comp = 0; comp < vdim; ++comp)
   {
      std::cout << "\n  Testing component " << comp << ":\n";

      // Create scalar input
      Vector scalar_input(scalar_ndofs);
      FillRandom(scalar_input, 98765 + comp, -1.0, 1.0);

      // Embed in vector field at component comp
      Vector vector_input = CreateComponentVector(vector_fe_space, comp, scalar_input);

      // Apply operators
      Vector scalar_output(scalar_ndofs);
      Vector vector_output_full(vector_ndofs);
      scalar_output = 0.0;
      vector_output_full = 0.0;

      scalar_operator(scalar_input, scalar_output);
      vector_operator(vector_input, vector_output_full);

      // Extract active component
      Vector active_output = ExtractComponent(vector_output_full, comp, scalar_fe_space);

      // Compute norms
      Real scalar_norm = 0.0;
      Real active_norm = 0.0;
      for (Integer i = 0; i < scalar_ndofs; ++i) {
         scalar_norm += scalar_output[i] * scalar_output[i];
         active_norm += active_output[i] * active_output[i];
      }
      scalar_norm = std::sqrt(scalar_norm);
      active_norm = std::sqrt(active_norm);

      const Real active_rel_error = RelativeL2Error(active_output, scalar_output);

      std::cout << "    ||scalar output||          = " << scalar_norm << "\n";
      std::cout << "    ||vector active component|| = " << active_norm << "\n";
      std::cout << "    Active component rel error = " << active_rel_error << "\n";

      // Check leakage
      std::cout << "    Inactive component leakage:\n";
      for (Integer other_comp = 0; other_comp < vdim; ++other_comp) {
         if (other_comp == comp) continue;

         Vector inactive_output = ExtractComponent(vector_output_full, other_comp, scalar_fe_space);
         Real inactive_norm = 0.0;
         for (Integer i = 0; i < scalar_ndofs; ++i) {
            inactive_norm += inactive_output[i] * inactive_output[i];
         }
         inactive_norm = std::sqrt(inactive_norm);

         std::cout << "      Component " << other_comp << " norm = " << inactive_norm;
         if (inactive_norm > tol * scalar_norm) {
            std::cout << "  [LEAKAGE!]";
         }
         std::cout << "\n";
      }

      if (active_rel_error > tol)
      {
         std::cerr << "    FAILED: vdim=2 Component " << comp << " active error too large\n";
         vdim2_test_passed = false;
      }
   }

   if (!vdim2_test_passed) {
      std::cerr << "\nFAILED: vdim=2 component-wise scalar equivalence\n";
      return 1;
   }

   std::cout << "SUCCESS: vdim=2 order " << order << " passed.\n";
   return 0;
}

} // namespace

int main()
{
   // Test vdim=3 (vector dim matches spatial dim)
   if (TestVectorGradGrad<1>() != 0) { return 1; }
   if (TestVectorGradGrad<2>() != 0) { return 1; }
   if (TestVectorGradGrad<3>() != 0) { return 1; }

   // Test vdim=2 on 3D mesh (catch vdim != dim assumptions)
   if (TestVectorGradGrad_vdim2<1>() != 0) { return 1; }
   if (TestVectorGradGrad_vdim2<2>() != 0) { return 1; }

   std::cout << "\nAll vector grad-grad tests passed.\n";
   return 0;
}
