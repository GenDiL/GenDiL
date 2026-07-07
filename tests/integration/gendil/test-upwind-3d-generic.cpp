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

template <typename VectorType>
void PrintVectorByElement(
   const VectorType& x,
   Integer num_elem,
   Integer elem_dofs,
   const char* label)
{
   std::cout << "\n" << label << " by element:\n";
   for (Integer e = 0; e < num_elem; ++e)
   {
      std::cout << "elem " << e << ": ";
      for (Integer i = 0; i < elem_dofs; ++i)
      {
         std::cout << x[e * elem_dofs + i];
         if (i + 1 != elem_dofs) { std::cout << ", "; }
      }
      std::cout << "\n";
   }
}

template <int Sign>
int TestFacetOnlyXDirection()
{
   static_assert(Sign == 1 || Sign == -1);

   std::cout << "\n=== 3D interior-facet-only upwind test, beta = ("
             << Sign << ",0,0) ===\n";

   // --------------------------------------------------------------------------
   // Mesh / FE space
   // --------------------------------------------------------------------------

   // Small mesh so the element-wise dumps stay readable.
   const Integer n = 3;
   const Real h = 1.0 / n;
   CartesianMesh<3> mesh({n, n, n}, {h, h, h}, {0.0, 0.0, 0.0});

   constexpr Integer order = 1;
   FiniteElementOrders<order, order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   const Integer elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem  = fe_space.GetNumberOfFiniteElements();
   const Integer num_dofs  = fe_space.GetNumberOfFiniteElementDofs();

   // --------------------------------------------------------------------------
   // Integration rule
   // --------------------------------------------------------------------------

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   // --------------------------------------------------------------------------
   // Input / output vectors
   // --------------------------------------------------------------------------

   Vector u_h(num_dofs);
   Vector v_h_generic(num_dofs);
   Vector v_h_matrix(num_dofs);

   u_h = 0.0;
   v_h_generic = 0.0;
   v_h_matrix  = 0.0;

   // Two useful choices:
   // 1) FillRandom(u_h);
   // 2) deterministic sparse input
   //
   // For debugging, deterministic is often easier to read.
   //
   // Here I activate one DOF in one interior element so the support is localized.
   // Adjust this if you want a different probing pattern.
   const Integer probe_element = 13; // middle-ish element in 3x3x3 lexicographic ordering
   const Integer probe_dof     = 0;  // first local dof
   u_h[probe_element * elem_dofs + probe_dof] = 1.0;

   // --------------------------------------------------------------------------
   // Constant x-direction velocity field
   // --------------------------------------------------------------------------

   constexpr Integer Dim = 3;

   auto beta_fn = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X)
      -> std::array<Real, Dim>
   {
      return { Real(Sign), Real(0.0), Real(0.0) };
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
   // Generic weak-form operator: facet term only
   // --------------------------------------------------------------------------

   TrialSpace<"displacement"> u_adv;
   TestSpace<"displacement"> v_adv;
   InteriorFacets<"mesh1"> interior_facets;

   auto beta = MakeVectorCoefficient<"beta", PhysicalCoordinate>(beta_fn);

   auto facet_only_wf =
      integrate(interior_facets, upwind(beta, u_adv) * jump(v_adv));

   auto wf_context = MakeWeakFormContext(
      MakeTrialField<"displacement">(fe_space),
      MakeIntegrationDomain<"mesh1">(fe_space)
   );

   auto generic_operator =
      MakeGenericOperator<KernelPolicy>(
         facet_only_wf,
         wf_context,
         integration_rule
      );

   generic_operator(u_h, v_h_generic);

   auto assembled_matrix =
      GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(
         facet_only_wf,
         wf_context,
         integration_rule
      );

   assembled_matrix(u_h, v_h_matrix);

   // --------------------------------------------------------------------------
   // Diagnostics
   // --------------------------------------------------------------------------

   std::cout << "||generic||   = " << L2Norm(v_h_generic) << "\n";
   std::cout << "||assembled|| = " << L2Norm(v_h_matrix)  << "\n";

   PrintComparison("generic", v_h_generic, "assembled", v_h_matrix);

   std::cout << "\nfull generic vector:\n"   << v_h_generic << "\n";
   std::cout << "\nfull assembled vector:\n" << v_h_matrix  << "\n";

   PrintVectorByElement(v_h_generic, num_elem, elem_dofs, "generic");
   PrintVectorByElement(v_h_matrix,  num_elem, elem_dofs, "assembled");

   const Real rel_err = RelativeL2Error(v_h_generic, v_h_matrix);
   const Real tol = 1e-12;

   if (rel_err > tol)
   {
      std::cerr << "FAILED: facet-only generic operator does not match assembly.\n";
      return 1;
   }

   std::cout << "SUCCESS.\n";
   return 0;
}

template <int Sign>
int TestFullAdvectionXDirection()
{
   static_assert(Sign == 1 || Sign == -1);

   std::cout << "\n=== 3D full advection test, beta = ("
             << Sign << ",0,0) ===\n";

   const Integer n = 3;
   const Real h = 1.0 / n;
   CartesianMesh<3> mesh({n, n, n}, {h, h, h}, {0.0, 0.0, 0.0});

   constexpr Integer order = 1;
   FiniteElementOrders<order, order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   const Integer elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem  = fe_space.GetNumberOfFiniteElements();
   const Integer num_dofs  = fe_space.GetNumberOfFiniteElementDofs();

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto integration_rule = MakeIntegrationRule(num_quads);

   Vector u_h(num_dofs);
   Vector v_h_legacy(num_dofs);
   Vector v_h_generic(num_dofs);
   Vector v_h_matrix(num_dofs);

   u_h = 0.0;
   v_h_legacy  = 0.0;
   v_h_generic = 0.0;
   v_h_matrix  = 0.0;

   const Integer probe_element = 13;
   const Integer probe_dof     = 0;
   u_h[probe_element * elem_dofs + probe_dof] = 1.0;

   constexpr Integer Dim = 3;

   auto beta_fn = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X)
      -> std::array<Real, Dim>
   {
      return { Real(Sign), Real(0.0), Real(0.0) };
   };

   auto adv = [beta_fn] GENDIL_HOST_DEVICE
      (const std::array<Real, Dim>& X, Real (&v)[Dim])
   {
      const auto beta = beta_fn(X);
      for (Integer d = 0; d < Dim; ++d)
      {
         v[d] = beta[d];
      }
   };

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy =
      ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto legacy_operator =
      MakeAdvectionOperator<KernelPolicy>(fe_space, integration_rule, adv);

   std::cout << "v_h legacy:\n" << v_h_legacy << "\n";
   legacy_operator(u_h, v_h_legacy);
   std::cout << "v_h legacy:\n" << v_h_legacy << "\n";

   TrialSpace<"displacement"> u_adv;
   TestSpace<"displacement"> v_adv;
   Cells<"mesh1"> cells;
   InteriorFacets<"mesh1"> interior_facets;

   auto beta = MakeVectorCoefficient<"beta", PhysicalCoordinate>(beta_fn);

   auto wf =
      integrate(cells, -u_adv * dot(beta, grad(v_adv)))
      + integrate(interior_facets, upwind(beta, u_adv) * jump(v_adv));

   auto wf_context = MakeWeakFormContext(
      MakeTrialField<"displacement">(fe_space),
      MakeIntegrationDomain<"mesh1">(fe_space)
   );

   auto generic_operator =
      MakeGenericOperator<KernelPolicy>(wf, wf_context, integration_rule);

   generic_operator(u_h, v_h_generic);

   auto assembled_matrix =
      GenericAssembly<MatrixAssemblyType::BSR, KernelPolicy>(wf, wf_context, integration_rule);

   assembled_matrix(u_h, v_h_matrix);

   std::cout << "||legacy||    = " << L2Norm(v_h_legacy)  << "\n";
   std::cout << "||generic||   = " << L2Norm(v_h_generic) << "\n";
   std::cout << "||assembled|| = " << L2Norm(v_h_matrix)  << "\n";

   PrintComparison("generic", v_h_generic, "legacy",    v_h_legacy);
   PrintComparison("generic", v_h_generic, "assembled", v_h_matrix);
   PrintComparison("legacy",  v_h_legacy,  "assembled", v_h_matrix);

   PrintVectorByElement(v_h_legacy,  num_elem, elem_dofs, "legacy");
   PrintVectorByElement(v_h_generic, num_elem, elem_dofs, "generic");
   PrintVectorByElement(v_h_matrix,  num_elem, elem_dofs, "assembled");

   return 0;
}

} // namespace

int main()
{
   CartesianMesh<3> mesh({3,3,3}, {1.0,1.0,1.0}, {0.0,0.0,0.0});

   for (GlobalIndex e = 0; e < mesh.GetNumberOfCells(); ++e)
   {
      auto idx = GetStructuredSubIndices(e, mesh.sizes);
      std::cout << "cell " << e
               << " -> (" << idx[0] << "," << idx[1] << "," << idx[2] << ")\n";

      auto xm = mesh.GetLocalFaceInfo(e, std::integral_constant<Integer,0>{});
      auto ym = mesh.GetLocalFaceInfo(e, std::integral_constant<Integer,1>{});
      auto zm = mesh.GetLocalFaceInfo(e, std::integral_constant<Integer,2>{});
      auto xp = mesh.GetLocalFaceInfo(e, std::integral_constant<Integer,3>{});
      auto yp = mesh.GetLocalFaceInfo(e, std::integral_constant<Integer,4>{});
      auto zp = mesh.GetLocalFaceInfo(e, std::integral_constant<Integer,5>{});

      std::cout << "  neighbors: "
               << xm.PlusSide().GetCellIndex() << ", "
               << ym.PlusSide().GetCellIndex() << ", "
               << zm.PlusSide().GetCellIndex() << ", "
               << xp.PlusSide().GetCellIndex() << ", "
               << yp.PlusSide().GetCellIndex() << ", "
               << zp.PlusSide().GetCellIndex() << "\n";
   }

   int status = 0;

   status |= TestFacetOnlyXDirection<+1>();
   status |= TestFacetOnlyXDirection<-1>();

   // Optional second stage:
   status |= TestFullAdvectionXDirection<+1>();
   status |= TestFullAdvectionXDirection<-1>();

   return status;
}
