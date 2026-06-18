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
int TestBoundaryMass()
{
   std::cout << "\n=== Boundary mass test, order = " << order << " ===\n";

   // --------------------------------------------------------------------------
   // Mesh / FE space
   // --------------------------------------------------------------------------

   const Integer n = 2;
   const Real h = 1.0 / n;

   // Non-periodic mesh to have boundary faces
   CartesianMesh<2> mesh({n, n}, {h, h}, {0.0, 0.0}, false);

   FiniteElementOrders<order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   // --------------------------------------------------------------------------
   // Integration rule
   // --------------------------------------------------------------------------

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d> num_quads;
   auto int_rules = MakeIntegrationRule(num_quads);

   // --------------------------------------------------------------------------
   // DOF vectors
   // --------------------------------------------------------------------------

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();

   Vector u_h(num_dofs);
   Vector v_h_generic(num_dofs);
   Vector v_h_matrix(num_dofs);

   // Initialize input with simple pattern
   u_h.WriteHostData();  // Mark buffer as valid before accessing elements
   for (Integer i = 0; i < num_dofs; ++i)
   {
      u_h[i] = 1.0 + 0.1 * i;
   }

   v_h_generic = 0.0;
   v_h_matrix = 0.0;

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
   // Generic boundary facet weak form
   // --------------------------------------------------------------------------

   TrialSpace<"displacement"> u;
   TestSpace<"displacement"> v;
   BoundaryFacets<"mesh1"> boundary_facets;

   // Simple boundary mass: integrate(boundary_facets, u * v)
   auto boundary_mass_wf = integrate(boundary_facets, u * v);

   auto boundary_wf_context = MakeWeakFormContext(
      MakeTrialField<"displacement">(fe_space),
      MakeIntegrationDomain<"mesh1">(fe_space)
   );

   auto generic_boundary_operator =
      MakeGenericOperator<KernelPolicy>(
         boundary_mass_wf,
         boundary_wf_context,
         int_rules
      );

   generic_boundary_operator(u_h, v_h_generic);

   // --------------------------------------------------------------------------
   // Generic assembled BSR matrix
   // --------------------------------------------------------------------------

   auto boundary_matrix = GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
      boundary_mass_wf,
      boundary_wf_context,
      int_rules
   );

   boundary_matrix(u_h, v_h_matrix);

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
      std::cerr << "ERROR: generic vs assembled relative L2 error "
                << err_generic_vs_assembled
                << " exceeds tolerance " << tol_generic_vs_assembled << "\n";
      return 1;
   }

   std::cout << "PASS: Boundary mass test for order " << order << "\n";
   return 0;
}

} // namespace

int main()
{
   int status = 0;
   status |= TestBoundaryMass<1>();
   status |= TestBoundaryMass<2>();
   status |= TestBoundaryMass<3>();
   return status;
}
