// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace mfem;
using namespace gendil;

void rotate_3d_vertices(int *v, int ref_face, int rot)
{
   std::vector<int> face_1, face_2;

   switch (ref_face/2)
   {
      case 0:
         face_1 = {v[0], v[1], v[2], v[3]};
         face_2 = {v[4], v[5], v[6], v[7]};
         break;
      case 1:
         face_1 = {v[1], v[5], v[6], v[2]};
         face_2 = {v[0], v[4], v[7], v[3]};
         break;
      case 2:
         face_1 = {v[4], v[5], v[1], v[0]};
         face_2 = {v[7], v[6], v[2], v[3]};
         break;
   }

   if (ref_face % 2 == 0)
   {
      std::reverse(face_1.begin(), face_1.end());
      std::reverse(face_2.begin(), face_2.end());
      std::swap(face_1, face_2);
   }

   std::rotate(face_1.begin(), face_1.begin() + rot, face_1.end());
   std::rotate(face_2.begin(), face_2.begin() + rot, face_2.end());

   for (int i = 0; i < 4; ++i)
   {
      v[i]   = face_1[i];
      v[i+4] = face_2[i];
   }
}

mfem::Mesh mesh_3d_orientation(int face_perm_1, int face_perm_2)
{
   static const int dim = 3;
   static const int nv = 12;
   static const int nel = 2;

   mfem::Mesh mesh(dim, nv, nel);

   double x[dim];

   x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0; x[1] = 0.0; x[2] = 0.0;
   mesh.AddVertex(x);

   x[0] = 0.0; x[1] = 1.0; x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0; x[1] = 1.0; x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0; x[1] = 1.0; x[2] = 0.0;
   mesh.AddVertex(x);

   x[0] = 0.0; x[1] = 0.0; x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0; x[1] = 0.0; x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0; x[1] = 0.0; x[2] = 1.0;
   mesh.AddVertex(x);

   x[0] = 0.0; x[1] = 1.0; x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0; x[1] = 1.0; x[2] = 1.0;
   mesh.AddVertex(x);

   // Non-Cartesian vertex to make the second hex geometrically nontrivial.
   x[0] = 3.0; x[1] = 1.0; x[2] = 2.0;
   mesh.AddVertex(x);

   int el[8];

   // Left hex.
   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   el[4] = 6;
   el[5] = 7;
   el[6] = 10;
   el[7] = 9;
   rotate_3d_vertices(el, face_perm_1/4, face_perm_1%4);
   mesh.AddHex(el);

   // Right/non-Cartesian hex.
   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   el[4] = 7;
   el[5] = 8;
   el[6] = 11;
   el[7] = 10;
   rotate_3d_vertices(el, face_perm_2/4, face_perm_2%4);
   mesh.AddHex(el);

   mesh.FinalizeHexMesh(true);
   mesh.GenerateBoundaryElements();
   mesh.Finalize();

   return mesh;
}

// -----------------------------------------------------------------------------
// Coefficients.
// Start with constant diffusivity. This isolates geometry, orientation,
// facet normals, and plus-side Jacobians. Once this passes, the asymmetric
// coefficient is useful for exposing PhysicalCoordinate mistakes.
// -----------------------------------------------------------------------------

template <Integer Dim>
struct ConstantDiffusivity
{
   GENDIL_HOST_DEVICE
   Real operator()(const std::array<Real, Dim>& X) const
   {
      return Real(1.0);
   }
};

template <Integer Dim>
struct AsymmetricDiffusivity
{
   GENDIL_HOST_DEVICE
   Real operator()(const std::array<Real, Dim>& X) const
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];

      return Real(2.0)
           + Real(0.11) * x
           + Real(0.07) * y
           + Real(0.13) * z
           + Real(0.03) * x * y
           + Real(0.05) * y * z
           + Real(0.02) * x * z;
   }
};

double ConstantDiffusivityMFEM(const mfem::Vector& X)
{
   return 1.0;
}

double AsymmetricDiffusivityMFEM(const mfem::Vector& X)
{
   const double x = X[0];
   const double y = X[1];
   const double z = X[2];

   return 2.0
        + 0.11 * x
        + 0.07 * y
        + 0.13 * z
        + 0.03 * x * y
        + 0.05 * y * z
        + 0.02 * x * z;
}

// -----------------------------------------------------------------------------
// Main generic SIPDG/MFEM comparison.
// -----------------------------------------------------------------------------

template <
   Integer order,
   Integer num_quad_1d,
   bool UseVariableCoefficient = false >
int test_generic_sipdg_orientation(int fp1, int fp2)
{
   constexpr Integer mesh_order = 1;
   constexpr Integer Dim = 3;

   mfem::Mesh non_periodic_mesh = mesh_3d_orientation(fp1, fp2);
   HexMesh< mesh_order > mesh = MakeHexMesh< mesh_order >(non_periodic_mesh);

   // --------------------------------------------------------------------------
   // GenDiL finite element space
   // --------------------------------------------------------------------------

   FiniteElementOrders< order, order, order > orders;

   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   IntegrationRuleNumPoints< num_quad_1d, num_quad_1d, num_quad_1d > num_quads;
   auto int_rules = MakeIntegrationRule(num_quads);

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy =
      ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   const Real sigma = Real(1.0);
   const Real kappa = 0.0;//Real((order + 1) * (order + 1));

   // --------------------------------------------------------------------------
   // Generic SIPDG weak form
   // --------------------------------------------------------------------------

   TrialSpace<"displacement"> u;
   TestSpace<"displacement"> v;

   Cells<"mesh1"> cells;
   InteriorFacets<"mesh1"> interior_facets;
   BoundaryFacets<"mesh1"> boundary_facets;

   auto diffusivity =
      [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X) -> Real
      {
         if constexpr (UseVariableCoefficient)
         {
            return AsymmetricDiffusivity<Dim>{}(X);
         }
         else
         {
            return ConstantDiffusivity<Dim>{}(X);
         }
      };

   auto mu = MakeCoefficient<"diffusivity", PhysicalCoordinate>(diffusivity);

   auto tau = MakeCoefficient<"penalty", InverseFacetSize>(
      [=] GENDIL_HOST_DEVICE (const Real& h_inv) -> Real
      {
         return kappa * h_inv;
      });

   auto diffusion_wf =
      integrate(cells, mu * dot(grad(u), grad(v)))
      + integrate(
            interior_facets,
            - average(mu * dot(grad(u), Normal{})) * jump(v)
            + sigma * jump(u) * average(mu * dot(grad(v), Normal{}))
            + tau * mu * jump(u) * jump(v)
         )
      + integrate(
            boundary_facets,
            - mu * dot(grad(u), Normal{}) * v
            + sigma * u * mu * dot(grad(v), Normal{})
            + tau * mu * u * v
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

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();

   mfem::Vector dofs_in(num_dofs);
   mfem::Vector dofs_out_generic(num_dofs);

   dofs_in.Randomize(1);
   generic_diffusion_operator(dofs_in, dofs_out_generic);

   mfem::Vector dofs_out_bsr(num_dofs);
   dofs_out_bsr = 0.0;

   auto generic_diffusion_matrix =
      GenericAssembly<KernelPolicy>(
         diffusion_wf,
         diffusion_wf_context,
         int_rules
      );

   generic_diffusion_matrix(dofs_in, dofs_out_bsr);

   // --------------------------------------------------------------------------
   // MFEM reference operator
   // --------------------------------------------------------------------------

   const int dim = non_periodic_mesh.Dimension();

   mfem::L2_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&non_periodic_mesh, &fec);

   mfem::GridFunction dofs_out_mfem(&fes);

   mfem::FunctionCoefficient diffusivity_c(
      UseVariableCoefficient
         ? AsymmetricDiffusivityMFEM
         : ConstantDiffusivityMFEM
   );

   constexpr int mfem_rule_order = 2 * (num_quad_1d - 1);

   const mfem::IntegrationRule& volume_ir =
      mfem::IntRules.Get(mfem::Geometry::CUBE, mfem_rule_order);

   const mfem::IntegrationRule& face_ir =
      mfem::IntRules.Get(mfem::Geometry::SQUARE, mfem_rule_order);

   mfem::BilinearForm blf(&fes);

   auto* volume_integrator =
      new mfem::DiffusionIntegrator(diffusivity_c);

   volume_integrator->SetIntegrationRule(volume_ir);

   auto* interior_face_integrator =
      new mfem::DGDiffusionIntegrator(diffusivity_c, sigma, kappa);

   interior_face_integrator->SetIntegrationRule(face_ir);

   blf.AddDomainIntegrator(volume_integrator);
   blf.AddInteriorFaceIntegrator(interior_face_integrator);

   auto* boundary_face_integrator =
      new mfem::DGDiffusionIntegrator(diffusivity_c, sigma, kappa);

   boundary_face_integrator->SetIntegrationRule(face_ir);

   blf.AddBdrFaceIntegrator(boundary_face_integrator);

   blf.Assemble();
   blf.Finalize();

   blf.SpMat().Mult(dofs_in, dofs_out_mfem);

   // --------------------------------------------------------------------------
   // Compare
   // --------------------------------------------------------------------------

   mfem::Vector diff_generic_mfem(dofs_out_generic);
   diff_generic_mfem -= dofs_out_mfem;

   mfem::Vector diff_bsr_mfem(dofs_out_bsr);
   diff_bsr_mfem -= dofs_out_mfem;

   mfem::Vector diff_bsr_generic(dofs_out_bsr);
   diff_bsr_generic -= dofs_out_generic;

   const Real abs_error_generic_mfem = diff_generic_mfem.Norml2();
   const Real abs_error_bsr_mfem = diff_bsr_mfem.Norml2();
   const Real abs_error_bsr_generic = diff_bsr_generic.Norml2();

   const Real ref_norm = dofs_out_mfem.Norml2();
   const Real gen_norm = dofs_out_generic.Norml2();
   const Real bsr_norm = dofs_out_bsr.Norml2();

   const Real rel_error_generic_mfem = abs_error_generic_mfem / ref_norm;
   const Real rel_error_bsr_mfem = abs_error_bsr_mfem / ref_norm;
   const Real rel_error_bsr_generic = abs_error_bsr_generic / gen_norm;

   std::cout << "TEST<" << fp1 << ", " << fp2 << ">"
            << " dofs/elem=" << num_elem_dofs
            << " elements=" << num_elem
            << " total_dofs=" << num_dofs
            << " |generic|=" << gen_norm
            << " |bsr|=" << bsr_norm
            << " |mfem|=" << ref_norm
            << " generic-mfem abs=" << abs_error_generic_mfem
            << " rel=" << rel_error_generic_mfem
            << " bsr-mfem abs=" << abs_error_bsr_mfem
            << " rel=" << rel_error_bsr_mfem
            << " bsr-generic abs=" << abs_error_bsr_generic
            << " rel=" << rel_error_bsr_generic
            << std::endl;

   const Real tolerance =
      UseVariableCoefficient ? Real(1e-8) : Real(1e-9);

   if (
      rel_error_generic_mfem > tolerance ||
      rel_error_bsr_mfem > tolerance ||
      rel_error_bsr_generic > tolerance )
   {
      std::cout << "FAILED generic SIPDG orientation test<"
               << fp1 << ", " << fp2 << ">"
               << " with errors:"
               << " generic-mfem=" << rel_error_generic_mfem
               << " bsr-mfem=" << rel_error_bsr_mfem
               << " bsr-generic=" << rel_error_bsr_generic
               << std::endl;

      // Uncomment when debugging a single failed orientation pair.
      /*
      std::cout << "\nInput vector:\n";
      dofs_in.Print(std::cout, num_elem_dofs);

      std::cout << "\nGenDiL matrix-free output:\n";
      dofs_out_generic.Print(std::cout, num_elem_dofs);

      std::cout << "\nGenDiL BSR output:\n";
      dofs_out_bsr.Print(std::cout, num_elem_dofs);

      std::cout << "\nMFEM output:\n";
      dofs_out_mfem.Print(std::cout, num_elem_dofs);

      std::cout << "\nBSR - matrix-free difference:\n";
      diff_bsr_generic.Print(std::cout, num_elem_dofs);

      std::cout << "\nMatrix-free - MFEM difference:\n";
      diff_generic_mfem.Print(std::cout, num_elem_dofs);

      std::cout << "\nBSR - MFEM difference:\n";
      diff_bsr_mfem.Print(std::cout, num_elem_dofs);
      */

      return 1;
   }

   return 0;
}

// -----------------------------------------------------------------------------
// Driver.
// -----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
#if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
#elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
#else
   const char device_config[] = "cpu";
#endif

   mfem::Device device(device_config);
   device.Print();

   constexpr Integer order = 1;
   constexpr Integer num_quad_1d = order + 2;

   // There are 6 possible hex reference faces and 4 rotations per face.
   constexpr size_t n_perms = 24;

   int failed = 0;

   // Start with constant coefficient. This is the important geometry/orientation
   // validation for the new plus-side Jacobian infrastructure.
   constexpr bool use_variable_coefficient = false;

   // For quick debugging, replace the loops by a single nontrivial pair:
   //
   //    failed += test_generic_sipdg_orientation
   //       <order, num_quad_1d, use_variable_coefficient>(fp1, fp2);
   //
   // where fp1/fp2 are the failing orientation IDs.

   for (size_t fp2 = 0; fp2 < n_perms; ++fp2)
   {
      for (size_t fp1 = 0; fp1 < n_perms; ++fp1)
      {
         failed += test_generic_sipdg_orientation
            <order, num_quad_1d, use_variable_coefficient>(
               static_cast<int>(fp1),
               static_cast<int>(fp2)
            );
      }
   }

   if (failed == 0)
   {
      std::cout << "SUCCESS: all generic SIPDG orientation tests passed."
                << std::endl;
      return 0;
   }

   std::cout << "FAILED: " << failed
             << " generic SIPDG orientation tests failed."
             << std::endl;

   return 1;
}
