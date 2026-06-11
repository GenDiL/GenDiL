// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Test diffusion operator with varying sigma parameter
// This tests SIPDG with sigma = -1 (NIPDG), 0 (IPDG), and 1 (SIPDG)
// If sigma=0 passes but sigma=±1 fail, it confirms the symmetry term is the issue

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
int TestDiffusionVarySigma(Real sigma, const char* sigma_name)
{
   std::cout << "\n=== SIPDG diffusion test with sigma = " << sigma
             << " (" << sigma_name << "), order = " << order << " ===\n";

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

   Vector v_h_legacy(num_dofs);
   Vector v_h_generic(num_dofs);
   Vector v_h_matrix(num_dofs);

   v_h_legacy  = 0.0;
   v_h_generic = 0.0;
   v_h_matrix  = 0.0;

   // --------------------------------------------------------------------------
   // Coefficients / legacy parameters
   // --------------------------------------------------------------------------

   constexpr Integer Dim = 3;

   auto velocity = [=] GENDIL_HOST_DEVICE (std::array<Real, Dim> const& X) -> Real
   {
      return 1.0;
   };

   const Real kappa = 0.0;
   const Real tau_value = kappa / h;

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
   // Legacy operator
   // --------------------------------------------------------------------------

   auto diffusion_operator =
      MakeDiffusionOperator<KernelPolicy>(
         fe_space,
         int_rules,
         velocity,
         sigma,
         kappa
      );

   diffusion_operator(u_h, v_h_legacy);

   // --------------------------------------------------------------------------
   // Generic SIPDG weak form
   // --------------------------------------------------------------------------

   TrialSpace<"displacement"> u;
   TestSpace<"displacement"> v;
   Cells<"mesh1"> cells;
   InteriorFacets<"mesh1"> interior_facets;

   auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(velocity);

   auto tau = MakeCoefficient<"tau", PhysicalCoordinate>(
      [=] GENDIL_HOST_DEVICE (const auto&) -> Real
      {
         return tau_value;
      });

   // SIPDG weak form with varying sigma
   auto diffusion_wf =
      integrate(cells, mu * dot(grad(u), grad(v)))
      + integrate(
            interior_facets,
            - average(mu * dot(grad(u), Normal{})) * jump(v)
            + tau * mu * jump(u) * jump(v)
         )
      + integrate(
            interior_facets,
            sigma * jump(u) * average(mu * dot(grad(v), Normal{}))
         );

   auto diffusion_wf_context = MakeWeakFormContext(
      MakeTrialField<"displacement">(fe_space),
      MakeDomain<"mesh1">(mesh)
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

   std::cout << "||legacy||   = " << L2Norm(v_h_legacy)  << "\n";
   std::cout << "||generic||  = " << L2Norm(v_h_generic) << "\n";
   std::cout << "||assembled||= " << L2Norm(v_h_matrix)  << "\n";

   PrintComparison("generic", v_h_generic, "legacy",    v_h_legacy);
   PrintComparison("generic", v_h_generic, "assembled", v_h_matrix);
   PrintComparison("legacy",  v_h_legacy,  "assembled", v_h_matrix);

   const Real tol_generic_vs_legacy    = 1e-10;
   const Real tol_generic_vs_assembled = 1e-10;
   const Real tol_legacy_vs_assembled  = 1e-10;

   const Real err_generic_vs_legacy =
      RelativeL2Error(v_h_generic, v_h_legacy);

   const Real err_generic_vs_assembled =
      RelativeL2Error(v_h_generic, v_h_matrix);

   const Real err_legacy_vs_assembled =
      RelativeL2Error(v_h_legacy, v_h_matrix);

   if (err_generic_vs_legacy > tol_generic_vs_legacy)
   {
      std::cerr << "FAILED: generic SIPDG diffusion does not match legacy diffusion (sigma=" << sigma << ").\n";
      return 1;
   }

   if (err_generic_vs_assembled > tol_generic_vs_assembled)
   {
      std::cerr << "FAILED: generic SIPDG diffusion does not match generic assembled matrix (sigma=" << sigma << ").\n";
      return 1;
   }

   if (err_legacy_vs_assembled > tol_legacy_vs_assembled)
   {
      std::cerr << "FAILED: legacy diffusion does not match generic assembled matrix (sigma=" << sigma << ").\n";
      return 1;
   }

   std::cout << "SUCCESS: sigma=" << sigma << ", order " << order << " passed.\n";
   return 0;
}

template <Integer order>
int TestAllSigmaValues()
{
   int failures = 0;

   // Test sigma = 0 (IPDG - no symmetry term)
   if (TestDiffusionVarySigma<order>(0.0, "IPDG") != 0) {
      std::cerr << "*** sigma=0 FAILED - volume + consistency terms have issues\n";
      failures++;
   }

   // Test sigma = 1 (SIPDG - symmetric)
   if (TestDiffusionVarySigma<order>(1.0, "SIPDG") != 0) {
      std::cerr << "*** sigma=1 FAILED\n";
      failures++;
   }

   // Test sigma = -1 (NIPDG - non-symmetric)
   if (TestDiffusionVarySigma<order>(-1.0, "NIPDG") != 0) {
      std::cerr << "*** sigma=-1 FAILED\n";
      failures++;
   }

   return failures;
}

} // namespace

int main()
{
   std::cout << "Testing diffusion operator with varying sigma parameter\n";
   std::cout << "sigma = -1: NIPDG (non-symmetric)\n";
   std::cout << "sigma =  0: IPDG (incomplete, no symmetry term)\n";
   std::cout << "sigma =  1: SIPDG (symmetric)\n\n";
   std::cout << "If sigma=0 passes but sigma=±1 fail, the symmetry term is the issue.\n\n";

   int total_failures = 0;

   total_failures += TestAllSigmaValues<1>();
   total_failures += TestAllSigmaValues<2>();
   total_failures += TestAllSigmaValues<3>();

   if (total_failures > 0) {
      std::cerr << "\n" << total_failures << " test(s) failed.\n";
      return 1;
   }

   std::cout << "\nAll vary-sigma diffusion tests passed.\n";
   return 0;
}
