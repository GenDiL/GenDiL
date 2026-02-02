// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>
#include <mfem.hpp>
#include <cmath>
#include <iostream>

using namespace gendil;

int main(int, char**)
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

   // ---- 2D mesh: 1 x Ny so all interior faces are y-normal ----
   const Integer nx = 1;
   const Integer ny = 4;
   const Real hx = 1.0 / nx;
   const Real hy = 1.0 / ny;
   Cartesian2DMesh mesh(hx, hy, nx, ny, {0.0, 0.0});

   // ---- FE space: Q1 L2 ----
   constexpr Integer order = 1;
   FiniteElementOrders<order, order> orders;
   auto fe  = MakeLobattoFiniteElement(orders);
   auto fes = MakeFiniteElementSpace(mesh, fe);

   // ---- Volume rule (face rules deduced internally) ----
   constexpr Integer nq1d = order + 2;
   IntegrationRuleNumPoints<nq1d, nq1d> nqs;
   auto int_rules = MakeIntegrationRule(nqs);

   // ---- Face mesh: interior faces only ----
   constexpr Integer Dim = decltype(fes)::Dim;
   static_assert(Dim == 2, "This test assumes 2D.");
   std::array<GlobalIndex, Dim> sizes{(GlobalIndex)nx, (GlobalIndex)ny};
   auto face_mesh = make_cartesian_interior_face_connectivity<Dim>(sizes);

   // ---- Advection a=(1,0); tangent to all faces (n=(0,±1)) -> zero flux ----
   auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real,Dim>&, Real(&v)[Dim]) {
      v[0] = 1.0; v[1] = 0.0;
   };

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<nq1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   // ---- Build and apply FACE operator (no BCs) ----
   auto face_op = MakeAdvectionFaceOperator<KernelPolicy>(fes, face_mesh, int_rules, adv);

   const Integer ndofs = fes.GetNumberOfFiniteElementDofs();
   mfem::Vector x(ndofs), y(ndofs);
   x.Randomize(1); // any input is fine; face flux should be zero

   face_op.Mult(x, y);

   // ---- Check: should be (near) zero ----
   y.HostRead();
   const double nrm = y.Norml2();
   std::cout << "||face_op*x||_2 = " << nrm << std::endl;
   const double tol = 1e-12;
   if (nrm > tol) {
      std::cerr << "FAIL: face contribution not zero for a·n=0 configuration.\n";
      return 1;
   }
   std::cout << "PASS\n";
   return 0;
}
