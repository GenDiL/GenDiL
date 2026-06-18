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

template <Integer order>
int TestMass()
{
   std::cout << "\n=== Mass regression test, order = " << order << " ===\n";

   // --------------------------------------------------------------------------
   // Mesh / FE space
   // --------------------------------------------------------------------------

   const Integer n = 6;
   const Real h = 1.0 / n;
   CartesianMesh<3> mesh({n, n, n}, {h, h, h}, {0.0, 0.0, 0.0});

   FiniteElementOrders<order, order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   // --------------------------------------------------------------------------
   // Integration rule
   // --------------------------------------------------------------------------

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   // --------------------------------------------------------------------------
   // Input / output vectors
   // --------------------------------------------------------------------------

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();

   Vector x(num_dofs);
   FillRandom(x);

   Vector y_legacy(num_dofs);
   Vector y_generic(num_dofs);
   Vector y_assembled(num_dofs);

   y_legacy    = 0.0;
   y_generic   = 0.0;
   y_assembled = 0.0;

   // --------------------------------------------------------------------------
   // Coefficient
   // --------------------------------------------------------------------------

   constexpr Integer Dim = 3;
   auto sigma = [=] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return 1.0 + x * y + z * z;
   };

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
   // Legacy operator
   // --------------------------------------------------------------------------

   auto legacy_mass_operator =
      MakeMassFiniteElementOperator<KernelPolicy>(fe_space, integration_rule, sigma);

   legacy_mass_operator(x, y_legacy);

   // --------------------------------------------------------------------------
   // Generic weak-form operator
   // --------------------------------------------------------------------------

   Cells<"mesh1"> domain;
   TrialSpace<"u"> u;
   TestSpace<"u"> v;

   auto rho = MakeCoefficient<"density", PhysicalCoordinate>(sigma);

   auto weak_form = integrate(domain, rho * u * v);

   auto weak_form_context = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeIntegrationDomain<"mesh1">(fe_space));

   auto generic_mass_operator =
      MakeGenericOperator<KernelPolicy>(
         weak_form,
         weak_form_context,
         integration_rule);

   generic_mass_operator(x, y_generic);

   // --------------------------------------------------------------------------
   // Generic assembled BSR matrix
   // --------------------------------------------------------------------------

   auto assembled_mass_matrix =
      GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
         weak_form,
         weak_form_context,
         integration_rule);

   assembled_mass_matrix(x, y_assembled);

   // --------------------------------------------------------------------------
   // Comparisons
   // --------------------------------------------------------------------------

   std::cout << "||legacy||    = " << L2Norm(y_legacy)    << "\n";
   std::cout << "||generic||   = " << L2Norm(y_generic)   << "\n";
   std::cout << "||assembled|| = " << L2Norm(y_assembled) << "\n";

   PrintComparison("generic", y_generic, "legacy", y_legacy);
   PrintComparison("generic", y_generic, "assembled", y_assembled);
   PrintComparison("legacy",  y_legacy,  "assembled", y_assembled);

   const Real tol_generic_vs_legacy    = 1e-12;
   const Real tol_generic_vs_assembled = 1e-12;
   const Real tol_legacy_vs_assembled  = 1e-12;

   const Real err_generic_vs_legacy =
      RelativeL2Error(y_generic, y_legacy);

   const Real err_generic_vs_assembled =
      RelativeL2Error(y_generic, y_assembled);

   const Real err_legacy_vs_assembled =
      RelativeL2Error(y_legacy, y_assembled);

   if (err_generic_vs_legacy > tol_generic_vs_legacy)
   {
      std::cerr << "FAILED: generic mass operator does not match legacy mass operator.\n";
      return 1;
   }

   if (err_generic_vs_assembled > tol_generic_vs_assembled)
   {
      std::cerr << "FAILED: generic mass operator does not match generic assembled matrix.\n";
      return 1;
   }

   if (err_legacy_vs_assembled > tol_legacy_vs_assembled)
   {
      std::cerr << "FAILED: legacy mass operator does not match generic assembled matrix.\n";
      return 1;
   }

   std::cout << "SUCCESS: order " << order << " passed.\n";
   return 0;
}

} // namespace

int main()
{
   if (TestMass<1>() != 0) { return 1; }
   if (TestMass<2>() != 0) { return 1; }
   if (TestMass<3>() != 0) { return 1; }

   std::cout << "\nAll mass regression tests passed.\n";
   return 0;
}
