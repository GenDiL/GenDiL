// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using namespace gendil;

namespace
{

void Rotate3DVertices(int *v, int ref_face, int rot)
{
   std::vector<int> face_1, face_2;

   switch (ref_face / 2)
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
      v[i] = face_1[i];
      v[i + 4] = face_2[i];
   }
}

mfem::Mesh MakeTwoHexNonCartesianMesh(int face_perm_1, int face_perm_2)
{
   constexpr int dim = 3;
   constexpr int nv = 12;
   constexpr int nel = 2;

   mfem::Mesh mesh(dim, nv, nel);
   double x[dim];

   x[0] = 0.0; x[1] = 0.0; x[2] = 0.0; mesh.AddVertex(x);
   x[0] = 1.0; x[1] = 0.0; x[2] = 0.0; mesh.AddVertex(x);
   x[0] = 2.0; x[1] = 0.0; x[2] = 0.0; mesh.AddVertex(x);
   x[0] = 0.0; x[1] = 1.0; x[2] = 0.0; mesh.AddVertex(x);
   x[0] = 1.0; x[1] = 1.0; x[2] = 0.0; mesh.AddVertex(x);
   x[0] = 2.0; x[1] = 1.0; x[2] = 0.0; mesh.AddVertex(x);
   x[0] = 0.0; x[1] = 0.0; x[2] = 1.0; mesh.AddVertex(x);
   x[0] = 1.0; x[1] = 0.0; x[2] = 1.0; mesh.AddVertex(x);
   x[0] = 2.0; x[1] = 0.0; x[2] = 1.0; mesh.AddVertex(x);
   x[0] = 0.0; x[1] = 1.0; x[2] = 1.0; mesh.AddVertex(x);
   x[0] = 1.0; x[1] = 1.0; x[2] = 1.0; mesh.AddVertex(x);

   // Skew one plus-side vertex so J_minus and J_plus differ on the shared face.
   x[0] = 3.0; x[1] = 1.0; x[2] = 2.0; mesh.AddVertex(x);

   int el[8];

   el[0] = 0; el[1] = 1; el[2] = 4;  el[3] = 3;
   el[4] = 6; el[5] = 7; el[6] = 10; el[7] = 9;
   Rotate3DVertices(el, face_perm_1 / 4, face_perm_1 % 4);
   mesh.AddHex(el);

   el[0] = 1; el[1] = 2; el[2] = 5;  el[3] = 4;
   el[4] = 7; el[5] = 8; el[6] = 11; el[7] = 10;
   Rotate3DVertices(el, face_perm_2 / 4, face_perm_2 % 4);
   mesh.AddHex(el);

   mesh.FinalizeHexMesh(true);
   mesh.GenerateBoundaryElements();
   mesh.Finalize();

   return mesh;
}

template <typename VectorType>
Real RelativeL2Error(const VectorType& a, const VectorType& b)
{
   GENDIL_VERIFY(a.Size() == b.Size(), "Vector sizes do not match.");

   Real err_sq = 0.0;
   Real norm_b_sq = 0.0;
   for (Integer i = 0; i < a.Size(); ++i)
   {
      const Real d = a[i] - b[i];
      err_sq += d * d;
      norm_b_sq += b[i] * b[i];
   }

   const Real abs_err = std::sqrt(err_sq);
   const Real norm_b = std::sqrt(norm_b_sq);
   return norm_b == 0.0 ? abs_err : abs_err / norm_b;
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

template <Integer order, Integer num_quad_1d>
int TestLegacyPlusJacobian(int fp1, int fp2)
{
   constexpr Integer mesh_order = 1;
   constexpr Integer Dim = 3;

   mfem::Mesh mfem_mesh = MakeTwoHexNonCartesianMesh(fp1, fp2);
   HexMesh<mesh_order> mesh = MakeHexMesh<mesh_order>(mfem_mesh);

   FiniteElementOrders<order, order, order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   IntegrationRuleNumPoints<num_quad_1d, num_quad_1d, num_quad_1d> num_quads;
   auto int_rules = MakeIntegrationRule(num_quads);

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy =
      ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto diffusivity = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>&) -> Real
   {
      return Real(1.0);
   };

   constexpr Real sigma = 0.0;
   constexpr Real kappa = 0.0;

   auto legacy_operator =
      MakeDiffusionOperator<KernelPolicy>(
         fe_space,
         int_rules,
         diffusivity,
         sigma,
         kappa);

   TrialSpace<"u"> u;
   TestSpace<"u"> v;
   Cells<"mesh"> cells;
   InteriorFacets<"mesh"> interior_facets;
   BoundaryFacets<"mesh"> boundary_facets;

   auto mu = MakeCoefficient<"mu", PhysicalCoordinate>(diffusivity);

   auto weak_form =
      integrate(cells, mu * dot(grad(u), grad(v)))
      + integrate(
           interior_facets,
           - average(mu * dot(grad(u), Normal{})) * jump(v))
      + integrate(
           boundary_facets,
           - mu * dot(grad(u), Normal{}) * v);

   auto weak_form_context = MakeWeakFormContext(
      MakeTrialField<"u">(fe_space),
      MakeDomain<"mesh">(mesh));

   auto generic_operator =
      MakeGenericOperator<KernelPolicy>(
         weak_form,
         weak_form_context,
         int_rules);

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   const Integer elem_dofs = finite_element.GetNumDofs();

   mfem::Vector dofs_in(num_dofs);
   mfem::Vector dofs_out_legacy(num_dofs);
   mfem::Vector dofs_out_generic(num_dofs);

   dofs_in = 0.0;
   dofs_out_legacy = 0.0;
   dofs_out_generic = 0.0;

   // Element 0 is zero. Element 1 is nonconstant so the element-0 interface
   // residual isolates the plus-side trial-gradient contribution.
   for (Integer i = 0; i < elem_dofs; ++i)
   {
      const Integer dx = i % 2;
      const Integer dy = (i / 2) % 2;
      const Integer dz = (i / 4) % 2;
      dofs_in[elem_dofs + i] =
         Real(1.0)
         + Real(0.31) * dx
         - Real(0.27) * dy
         + Real(0.53) * dz
         + Real(0.19) * dx * dy
         - Real(0.11) * dy * dz
         + Real(0.07) * dx * dz;
   }

   legacy_operator(dofs_in, dofs_out_legacy);
   generic_operator(dofs_in, dofs_out_generic);

   const Real rel_error = RelativeL2Error(dofs_out_legacy, dofs_out_generic);

   std::cout << "TEST<" << fp1 << ", " << fp2 << ">"
             << " |legacy|=" << L2Norm(dofs_out_legacy)
             << " |generic|=" << L2Norm(dofs_out_generic)
             << " legacy-generic rel=" << rel_error
             << std::endl;

   constexpr Real tolerance = 1e-10;
   if (rel_error > tolerance)
   {
      std::cerr << "FAILED: legacy diffusion does not match generic SIPDG "
                << "for plus-side Jacobian regression." << std::endl;
      return 1;
   }

   return 0;
}

} // namespace

int main()
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

   const std::array<std::pair<int, int>, 4> orientation_cases{
      std::pair<int, int>{0, 0},
      std::pair<int, int>{0, 7},
      std::pair<int, int>{11, 3},
      std::pair<int, int>{17, 22}
   };

   int failed = 0;
   for (const auto& [fp1, fp2] : orientation_cases)
   {
      failed += TestLegacyPlusJacobian<order, num_quad_1d>(fp1, fp2);
   }

   if (failed == 0)
   {
      std::cout << "SUCCESS: legacy plus-side Jacobian regression passed."
                << std::endl;
      return 0;
   }

   std::cout << "FAILED: " << failed
             << " legacy plus-side Jacobian regression cases failed."
             << std::endl;
   return 1;
}
