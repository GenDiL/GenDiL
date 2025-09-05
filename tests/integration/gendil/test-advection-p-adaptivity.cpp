// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

using namespace gendil;

int main(int, char**)
{
   // --------------------------
   // Two side-by-side Cartesian meshes in 3D
   // Left: [0,1]^3, Right: [1,2]×[0,1]^2 (same h, geometry-conforming)
   // --------------------------
   const Integer num_elem_1d = 3;
   const Real    h  = 1.0 / num_elem_1d;

   const Integer nxL = num_elem_1d, ny = num_elem_1d, nz = num_elem_1d;
   const Integer nxR = num_elem_1d; // p-nonconforming only → same tangential partition

   Point<3> originL = {0.0, 0.0, 0.0};
   Point<3> originR = {1.0, 0.0, 0.0}; // shift by +1 in x

   Cartesian3DMesh meshL(h, h, h, nxL, ny, nz, originL);
   Cartesian3DMesh meshR(h, h, h, nxR, ny, nz, originR);

   // --------------------------
   // FE spaces with different polynomial orders (p-nonconforming)
   // --------------------------
   constexpr Integer pL = 1;
   constexpr Integer pR = 3;

   auto feL = MakeLobattoFiniteElement(FiniteElementOrders<pL,pL,pL>{});
   auto feR = MakeLobattoFiniteElement(FiniteElementOrders<pR,pR,pR>{});
   

   L2Restriction resL{ 0 };
   auto fe_space_L = MakeFiniteElementSpace(meshL, feL, resL);
   const Integer ndofsL = fe_space_L.GetNumberOfFiniteElementDofs();
   L2Restriction resR{ ndofsL };
   auto fe_space_R = MakeFiniteElementSpace(meshR, feR, resR);
   const Integer ndofsR = fe_space_R.GetNumberOfFiniteElementDofs();

   // Dim helper
   constexpr Integer Dim = GetDim(fe_space_L);
   static_assert(Dim == 3, "This test is set up for 3D.");

   // --------------------------
   // Volume integration rule (factory will deduce face rules)
   // --------------------------
   constexpr Integer q1d = (pL > pR ? pL : pR) + 2;
   auto int_rules = MakeIntegrationRule(IntegrationRuleNumPoints<q1d,q1d,q1d>{});

   // --------------------------
   // Face mesh: couples right faces (Axis=0, +x) of left mesh
   //            with left faces (Axis=0, -x) of right mesh
   // --------------------------
   std::array<GlobalIndex, Dim> sizesL{(GlobalIndex)nxL, (GlobalIndex)ny, (GlobalIndex)nz};
   std::array<GlobalIndex, Dim> sizesR{(GlobalIndex)nxR, (GlobalIndex)ny, (GlobalIndex)nz};

   CartesianIntermeshFaceConnectivity<Dim, 0> face_mesh(sizesL, sizesR);

   // --------------------------
   // Advection field
   // --------------------------
   auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X, Real (&v)[Dim]) {
      const Real x = X[0], y = X[1], z = X[2];
      v[0] = y; v[1] = z; v[2] = x;
   };

   // --------------------------
   // Kernel policy
   // --------------------------
#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout  = ThreadBlockLayout<q1d,q1d,q1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   // --------------------------
   // Build the p-nonconforming FACE operator and apply once
   // --------------------------
   auto face_op = MakeAdvectionFaceOperator<KernelPolicy>(
                     fe_space_L, fe_space_R, face_mesh, int_rules, adv);

   Vector u(ndofsL + ndofsR), r(ndofsL + ndofsR);
   u = 1.0; r = 0.0;

   face_op(u, r); // compile/link + execution sanity check

   // --------------------------
   // Optional: conservation sanity (constant state → net flux ~ 0)
   // --------------------------
   const Real *r_data = r.ReadHostData();
   Real sumL = 0.0, sumR = 0.0;
   for (Integer i = 0; i < ndofsL; ++i) sumL += r_data[i];
   for (Integer j = ndofsL; j < ndofsL + ndofsR; ++j) sumR += r_data[j];

   if (std::abs(sumL + sumR) > 1e-10) return 2;

   return 0;
}
