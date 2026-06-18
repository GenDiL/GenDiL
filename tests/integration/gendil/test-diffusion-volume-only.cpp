// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Test diffusion operator with VOLUME TERM ONLY
// This isolates the volume integral: ∫_Ω μ ∇u · ∇v dx
// No face terms (consistency, penalty, or symmetry)

#include <gendil/gendil.hpp>

#include <array>
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

template <Integer order>
int TestDiffusionVolumeOnly()
{
   std::cout << "\n=== Diffusion VOLUME TERM ONLY test, order = "
             << order << " ===\n";

   // --------------------------------------------------------------------------
   // Mesh / FE space
   // --------------------------------------------------------------------------

   const Integer n = 2;
   const Real h = 1.0 / n;

   // Periodic mesh
   CartesianMesh<3> mesh({n, n, n}, {h, h, h}, {0.0, 0.0, 0.0}, true);

   FiniteElementOrders<order, order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   // --------------------------------------------------------------------------
   // Integration rule
   // --------------------------------------------------------------------------

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto int_rules = MakeIntegrationRule(num_quads);

   // --------------------------------------------------------------------------
   // Input / output vectors
   // --------------------------------------------------------------------------

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();

   Vector u_h(num_dofs);
   FillRandom(u_h);

   Vector v_h_generic(num_dofs);
   Vector v_h_matrix(num_dofs);

   v_h_generic = 0.0;
   v_h_matrix  = 0.0;

   // --------------------------------------------------------------------------
   // Coefficients
   // --------------------------------------------------------------------------

   constexpr Integer Dim = 3;

   auto velocity = [=] GENDIL_HOST_DEVICE (std::array<Real, Dim> const& X) -> Real
   {
      return 1.0;
   };

   // --------------------------------------------------------------------------
   // Kernel policy
   // --------------------------------------------------------------------------

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy =
      ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   // --------------------------------------------------------------------------
   // Generic VOLUME TERM ONLY weak form
   // --------------------------------------------------------------------------

   TrialSpace<"displacement"> u;
   TestSpace<"displacement"> v;
   Cells<"mesh1"> cells;

   auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(velocity);

   // ONLY the volume term - no face integrals
   auto diffusion_wf = integrate(cells, mu * dot(grad(u), grad(v)));

   auto diffusion_wf_context = MakeWeakFormContext(
      MakeTrialField<"displacement">(fe_space),
      MakeIntegrationDomain<"mesh1">(fe_space)
   );

   auto generic_diffusion_operator =
      MakeGenericOperator<KernelPolicy>(
         diffusion_wf,
         diffusion_wf_context,
         int_rules
      );

   generic_diffusion_operator(u_h, v_h_generic);

   // --------------------------------------------------------------------------
   // Generic assembled BSR matrix
   // --------------------------------------------------------------------------

   auto diffusion_matrix = GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
      diffusion_wf,
      diffusion_wf_context,
      int_rules
   );

   diffusion_matrix(u_h, v_h_matrix);

   // --------------------------------------------------------------------------
   // Comparisons
   // --------------------------------------------------------------------------

   std::cout << "||generic||  = " << L2Norm(v_h_generic) << "\n";
   std::cout << "||assembled||= " << L2Norm(v_h_matrix)  << "\n";

   PrintComparison("generic", v_h_generic, "assembled", v_h_matrix);

   const Real tol_generic_vs_assembled = 1e-10;

   const Real err_generic_vs_assembled =
      RelativeL2Error(v_h_generic, v_h_matrix);

   if (err_generic_vs_assembled > tol_generic_vs_assembled)
   {
      std::cerr << "FAILED: generic volume-only diffusion does not match assembled matrix.\n";
      std::cerr << "\nv_h generic: " << v_h_generic << "\n";
      std::cerr << "\nv_h assembled:  " << v_h_matrix << "\n";
      return 1;
   }

   std::cout << "SUCCESS: volume-only test, order " << order << " passed.\n";
   return 0;
}

} // namespace

int main()
{
   std::cout << "Testing diffusion operator with VOLUME TERM ONLY\n";
   std::cout << "This validates: ∫_Ω μ ∇u · ∇v dx\n";

   if (TestDiffusionVolumeOnly<1>() != 0) { return 1; }
   if (TestDiffusionVolumeOnly<2>() != 0) { return 1; }
   if (TestDiffusionVolumeOnly<3>() != 0) { return 1; }

   std::cout << "\nAll volume-only diffusion tests passed.\n";
   return 0;
}
