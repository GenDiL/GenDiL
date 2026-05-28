// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>

using namespace gendil;

int main(int, char**)
{
   // Two side-by-side Cartesian meshes in 3D.
   // Left: [0,1]^3, right: [1,2]x[0,1]^2.
   const Integer num_elem_1d = 3;
   const Real h = 1.0 / num_elem_1d;

   const Integer nxL = num_elem_1d;
   const Integer ny = num_elem_1d;
   const Integer nz = num_elem_1d;
   const Integer nxR = num_elem_1d;

   Point<3> originL = {0.0, 0.0, 0.0};
   Point<3> originR = {1.0, 0.0, 0.0};

   Cartesian3DMesh meshL(h, h, h, nxL, ny, nz, originL);
   Cartesian3DMesh meshR(h, h, h, nxR, ny, nz, originR);

   // FE spaces with different polynomial orders (p-nonconforming).
   constexpr Integer pL = 1;
   constexpr Integer pR = 3;

   auto feL = MakeLobattoFiniteElement(FiniteElementOrders<pL, pL, pL>{});
   auto feR = MakeLobattoFiniteElement(FiniteElementOrders<pR, pR, pR>{});

   L2Restriction resL{0};
   auto fe_space_L = MakeFiniteElementSpace(meshL, feL, resL);
   const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();

   L2Restriction resR{ndofsL};
   auto fe_space_R = MakeFiniteElementSpace(meshR, feR, resR);
   const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();

   constexpr Integer Dim = GetDim(fe_space_L);
   static_assert(Dim == 3, "This test is set up for 3D.");

   constexpr Integer q1d = (pL > pR ? pL : pR) + 2;
   auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d, q1d, q1d>{});

   std::array<GlobalIndex, Dim> sizesL{
      (GlobalIndex)nxL,
      (GlobalIndex)ny,
      (GlobalIndex)nz};
   std::array<GlobalIndex, Dim> sizesR{
      (GlobalIndex)nxR,
      (GlobalIndex)ny,
      (GlobalIndex)nz};

   // Connect the +x face of the left mesh to the -x face of the right mesh.
   constexpr Integer Axis = 0;
   constexpr Integer LFI = Axis + Dim;
   CartesianIntermeshFaceConnectivity<Dim, LFI> face_mesh(sizesL, sizesR);

   auto adv =
      [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X, Real (&v)[Dim])
      {
         const Real x = X[0];
         const Real y = X[1];
         const Real z = X[2];
         v[0] = y;
         v[1] = z;
         v[2] = x;
      };

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<q1d, q1d, q1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy =
      ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto face_op = MakeAdvectionFaceOperator<KernelPolicy>(
      fe_space_L, fe_space_R, face_mesh, int_rules, adv);

   Vector u(ndofsL + ndofsR);
   Vector r(ndofsL + ndofsR);
   u = 1.0;
   r = 0.0;

   face_op(u, r);

   const Real *r_data = r.ReadHostData();
   Real sumL = 0.0;
   Real sumR = 0.0;
   for (Integer i = 0; i < ndofsL; ++i)
   {
      sumL += r_data[i];
   }
   for (Integer i = ndofsL; i < ndofsL + ndofsR; ++i)
   {
      sumR += r_data[i];
   }

   if (std::abs(sumL + sumR) > 1e-10)
   {
      return 2;
   }

   return 0;
}
