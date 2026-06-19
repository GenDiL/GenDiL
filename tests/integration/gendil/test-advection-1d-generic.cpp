// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

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
int TestAdvection1D()
{
   std::cout << "\n=== 1D advection regression test, order = " << order << " ===\n";

   // --------------------------------------------------------------------------
   // Mesh / FE space
   // --------------------------------------------------------------------------

   const Integer n = 8;
   const Real h = 1.0 / n;
   Cartesian1DMesh mesh(h, n);

   FiniteElementOrders<order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   // --------------------------------------------------------------------------
   // Integration rule
   // --------------------------------------------------------------------------

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints<num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

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
   // Velocity field
   // --------------------------------------------------------------------------

   constexpr Integer Dim = 1;

   auto beta_fn = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X)
      -> std::array<Real, Dim>
   {
      const Real x = X[0];
      return { x };
      // alternative:
      // return { x * (1.0 - x) };
   };

   auto adv = [beta_fn] GENDIL_HOST_DEVICE
      (const std::array<Real, Dim>& X, Real (&v)[Dim])
   {
      const auto beta = beta_fn(X);
      v[0] = beta[0];
   };

   // --------------------------------------------------------------------------
   // Kernel policy
   // --------------------------------------------------------------------------

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy =
      ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   // --------------------------------------------------------------------------
   // Legacy operator (no boundary term)
   // --------------------------------------------------------------------------

   auto legacy_advection_operator =
      MakeAdvectionOperator<KernelPolicy>(fe_space, integration_rule, adv);

   legacy_advection_operator(u_h, v_h_legacy);

   // --------------------------------------------------------------------------
   // Generic weak-form operator
   // --------------------------------------------------------------------------

   TrialSpace<"displacement"> u_adv;
   TestSpace<"displacement"> v_adv;
   Cells<"mesh1"> cells;
   InteriorFacets<"mesh1"> interior_facets;

   auto beta = MakeVectorCoefficient<"beta", PhysicalCoordinate>(beta_fn);

   auto advection_dg_wf =
      integrate(cells, -u_adv * dot(beta, grad(v_adv)))
      + integrate(interior_facets, upwind(beta, u_adv) * jump(v_adv));

   auto advection_wf_context = MakeWeakFormContext(
      MakeTrialField<"displacement">(fe_space),
      MakeIntegrationDomain<"mesh1">(fe_space)
   );

   auto generic_advection_operator =
      MakeGenericOperator<KernelPolicy>(
         advection_dg_wf,
         advection_wf_context,
         integration_rule
      );

   generic_advection_operator(u_h, v_h_generic);

   // --------------------------------------------------------------------------
   // Generic assembled BSR matrix
   // --------------------------------------------------------------------------

   auto advection_matrix = GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
      advection_dg_wf,
      advection_wf_context,
      integration_rule
   );

   advection_matrix(u_h, v_h_matrix);

   // --------------------------------------------------------------------------
   // Comparisons
   // --------------------------------------------------------------------------

   std::cout << "||legacy||   = " << L2Norm(v_h_legacy)  << "\n";
   std::cout << "||generic||  = " << L2Norm(v_h_generic) << "\n";
   std::cout << "||assembled||= " << L2Norm(v_h_matrix)  << "\n";

   PrintComparison("generic", v_h_generic, "legacy",    v_h_legacy);
   PrintComparison("generic", v_h_generic, "assembled", v_h_matrix);
   PrintComparison("legacy",  v_h_legacy,  "assembled", v_h_matrix);

   const Real tol_generic_vs_legacy    = 1e-12;
   const Real tol_generic_vs_assembled = 1e-12;
   const Real tol_legacy_vs_assembled  = 1e-12;

   const Real err_generic_vs_legacy =
      RelativeL2Error(v_h_generic, v_h_legacy);

   const Real err_generic_vs_assembled =
      RelativeL2Error(v_h_generic, v_h_matrix);

   const Real err_legacy_vs_assembled =
      RelativeL2Error(v_h_legacy, v_h_matrix);

   if (err_generic_vs_legacy > tol_generic_vs_legacy)
   {
      std::cerr << "FAILED: generic advection operator does not match legacy advection operator.\n";
      return 1;
   }

   if (err_generic_vs_assembled > tol_generic_vs_assembled)
   {
      std::cerr << "FAILED: generic advection operator does not match generic assembled matrix.\n";
      return 1;
   }

   if (err_legacy_vs_assembled > tol_legacy_vs_assembled)
   {
      std::cerr << "FAILED: legacy advection operator does not match generic assembled matrix.\n";
      return 1;
   }

   std::cout << "SUCCESS: order " << order << " passed.\n";
   return 0;
}

} // namespace

int main()
{
   if (TestAdvection1D<1>() != 0) { return 1; }
   if (TestAdvection1D<2>() != 0) { return 1; }
   if (TestAdvection1D<3>() != 0) { return 1; }

   std::cout << "\nAll 1D advection regression tests passed.\n";
   return 0;
}
