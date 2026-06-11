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

   Real norm_b_sq = 0.0;
   for (Integer i = 0; i < b.Size(); ++i)
   {
      norm_b_sq += b[i] * b[i];
   }

   const Real norm_b = std::sqrt(norm_b_sq);
   if (norm_b == 0.0)
   {
      return abs_err;
   }
   return abs_err / norm_b;
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
void PrintComparison(
   const char* label_a,
   const VectorType& a,
   const char* label_b,
   const VectorType& b)
{
   std::cout << label_a << " vs " << label_b
             << " | abs L2 error = " << AbsoluteL2Error(a, b)
             << ", rel L2 error = " << RelativeL2Error(a, b)
             << "\n";
}

// Component-major layout helpers
// Layout: [All dofs for comp 0][All dofs for comp 1][All dofs for comp 2]
// This matches the layout used by VectorShapeFunctions and ReadDofs/WriteDofs

template<typename FiniteElementSpace>
Vector CreateComponentVector(
   const FiniteElementSpace& vector_fe_space,
   Integer component,
   const Vector& scalar_input)
{
   // Get scalar dof count from one component
   constexpr Integer vdim = FiniteElementSpace::finite_element_type::shape_functions::num_comp;
   const Integer scalar_ndofs = scalar_input.Size();

   Vector result(scalar_ndofs * vdim);
   result = 0.0;

   // Component-major layout: component c starts at offset c * scalar_ndofs
   for (Integer i = 0; i < scalar_ndofs; ++i) {
      result[component * scalar_ndofs + i] = scalar_input[i];
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

   // Need to mark result as valid before writing via operator[]
   Real* result_ptr = result.WriteHostData();
   const Real* output_ptr = vector_output.ReadHostData();

   // Component-major layout: component c starts at offset c * scalar_ndofs
   for (Integer i = 0; i < scalar_ndofs; ++i) {
      result_ptr[i] = output_ptr[component * scalar_ndofs + i];
   }
   return result;
}

template <Integer order>
int TestVectorMass()
{
   std::cout << "\n=== Vector mass test, order = " << order << " ===\n";

   // --------------------------------------------------------------------------
   // Mesh / FE space
   // --------------------------------------------------------------------------

   const Integer n = 6;
   const Real h = 1.0 / n;
   CartesianMesh<3> mesh({n, n, n}, {h, h, h}, {0.0, 0.0, 0.0});

   FiniteElementOrders<order, order, order> orders;
   auto scalar_finite_element = MakeLobattoFiniteElement(orders);

   // Create vector FE space with vdim = 3
   auto vector_finite_element = MakeVectorFiniteElement(
      scalar_finite_element,
      scalar_finite_element,
      scalar_finite_element);

   auto vector_fe_space = MakeFiniteElementSpace(mesh, vector_finite_element);
   auto scalar_fe_space = MakeFiniteElementSpace(mesh, scalar_finite_element);

   // --------------------------------------------------------------------------
   // Integration rule
   // --------------------------------------------------------------------------

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   // --------------------------------------------------------------------------
   // Kernel policy
   // --------------------------------------------------------------------------

#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 2;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   // --------------------------------------------------------------------------
   // Coefficient
   // --------------------------------------------------------------------------

   constexpr Integer Dim = 3;
   auto rho_fn = [=] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return 1.0 + x * y + z * z;
   };

   // --------------------------------------------------------------------------
   // Generic weak-form vector operator
   // --------------------------------------------------------------------------

   Cells<"mesh1"> domain;
   VectorTrialSpace<"u"> u;
   VectorTestSpace<"u"> v;

   auto rho = MakeCoefficient<"density", PhysicalCoordinate>(rho_fn);

   auto weak_form = integrate(domain, rho * dot(u, v));

   auto weak_form_context = MakeWeakFormContext(
      MakeTrialField<"u">(vector_fe_space),  // Vector shape comes from FE space
      MakeDomain<"mesh1">(mesh));

   auto vector_mass_operator =
      MakeGenericOperator<KernelPolicy>(
         weak_form,
         weak_form_context,
         integration_rule);

   // --------------------------------------------------------------------------
   // Oracle 1: Constant-field energy
   // --------------------------------------------------------------------------

   const Integer vector_ndofs = vector_fe_space.GetNumberOfFiniteElementDofs();
   const Integer scalar_ndofs = scalar_fe_space.GetNumberOfFiniteElementDofs();

   constexpr Integer vdim = decltype(vector_finite_element)::shape_functions::num_comp;

   std::cout << "Vector dofs: " << vector_ndofs
             << ", Scalar dofs: " << scalar_ndofs
             << ", vdim: " << vdim << "\n";

   // Constant field: all dofs = 1.0
   Vector constant_field(vector_ndofs);
   constant_field = 1.0;

   Vector output_constant(vector_ndofs);
   output_constant = 0.0;

   vector_mass_operator(constant_field, output_constant);

   const Real constant_energy = Dot(constant_field, output_constant);

   // Expected: ∫ ρ dV * vdim
   // Volume = 1.0, ρ average ≈ 1.33, vdim = 3
   // Rough estimate for validation
   const Real volume = 1.0;

   std::cout << "Oracle 1 - Constant field energy:\n";
   std::cout << "  Energy = " << constant_energy << "\n";
   std::cout << "  Expected order of magnitude: volume * avg(rho) * vdim = "
             << volume << " * ~1.3 * " << vdim << " ≈ 3.9\n";

   if (std::abs(constant_energy) < 1e-10 || constant_energy < 0.0)
   {
      std::cerr << "FAILED: Constant field energy is invalid.\n";
      return 1;
   }

   // --------------------------------------------------------------------------
   // Oracle 2: Component-wise scalar equivalence
   // --------------------------------------------------------------------------

   std::cout << "\nOracle 2 - Component-wise scalar equivalence:\n";

   // Create scalar mass operator with same weak form
   TrialSpace<"u"> u_scalar;
   TestSpace<"u"> v_scalar;
   auto scalar_weak_form = integrate(domain, rho * u_scalar * v_scalar);

   auto scalar_weak_form_context = MakeWeakFormContext(
      MakeTrialField<"u">(scalar_fe_space),
      MakeDomain<"mesh1">(mesh));

   auto scalar_mass_operator =
      MakeGenericOperator<KernelPolicy>(
         scalar_weak_form,
         scalar_weak_form_context,
         integration_rule);

   // Test each component
   for (Integer comp = 0; comp < vdim; ++comp)
   {
      // Create scalar input
      Vector scalar_input(scalar_ndofs);
      FillRandom(scalar_input);

      // Embed in vector field at component comp
      Vector vector_input = CreateComponentVector(vector_fe_space, comp, scalar_input);

      // Apply operators
      Vector scalar_output(scalar_ndofs);
      Vector vector_output(vector_ndofs);
      scalar_output = 0.0;
      vector_output = 0.0;

      scalar_mass_operator(scalar_input, scalar_output);
      vector_mass_operator(vector_input, vector_output);

      // Extract component from vector output
      Vector extracted_output = ExtractComponent(vector_output, comp, scalar_fe_space);

      // Compare
      const Real rel_error = RelativeL2Error(extracted_output, scalar_output);
      std::cout << "  Component " << comp << ": rel error = " << rel_error << "\n";

      const Real tol = 1e-12;
      if (rel_error > tol)
      {
         std::cerr << "FAILED: Component " << comp
                   << " does not match scalar operator (error = " << rel_error << ").\n";
         return 1;
      }
   }

   std::cout << "SUCCESS: order " << order << " passed both oracles.\n";
   return 0;
}

} // namespace

int main()
{
   if (TestVectorMass<1>() != 0) { return 1; }
   if (TestVectorMass<2>() != 0) { return 1; }
#ifndef GENDIL_USE_CUDA
   if (TestVectorMass<3>() != 0) { return 1; }
#endif

   std::cout << "\nAll vector mass tests passed.\n";
   return 0;
}
