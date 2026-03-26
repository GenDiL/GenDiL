// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

using namespace gendil;

int main(int argc, char* argv[])
{
   const Integer num_elem_1d = 3;
   const Real h = 1.0 / num_elem_1d;

   const Integer nx = num_elem_1d;
   const Integer ny = num_elem_1d;
   const Integer nz = num_elem_1d;

   Point<3> origin = {0.0, 0.0, 0.0};
   Cartesian3DMesh mesh(h, h, h, nx, ny, nz, origin);

   // --------------------------
   // Finite element space (Q1 L2)
   // --------------------------
   constexpr Integer order = 1;
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders<order_X, order_Y, order_Z> orders;

   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space       = MakeFiniteElementSpace(mesh, finite_element);

   // --------------------------
   // Volume integration rule (used to deduce face rules internally)
   // --------------------------
   constexpr Integer num_quad_1d = order + 2;
   constexpr Integer num_quad_X = num_quad_1d;
   constexpr Integer num_quad_Y = num_quad_1d;
   constexpr Integer num_quad_Z = num_quad_1d;
   IntegrationRuleNumPoints<num_quad_X, num_quad_Y, num_quad_Z> num_quads;
   auto int_rules = MakeIntegrationRule(num_quads);

   // --------------------------
   // Face mesh (interior faces on the Cartesian grid)
   // --------------------------
   // The helper returns a face connectivity for interior faces in each direction.
   constexpr Integer Dim = GetDim( fe_space );
   static_assert(Dim == 3, "This test is set up for 3D. Switch mesh for 2D runs.");

   std::array<GlobalIndex, Dim> sizes{(GlobalIndex)nx, (GlobalIndex)ny, (GlobalIndex)nz};
   auto face_mesh = make_cartesian_interior_face_connectivity<Dim>(sizes);

   // --------------------------
   // Advection field and (placeholder) boundary field
   // --------------------------
   auto adv = [] GENDIL_HOST_DEVICE (const std::array<Real, Dim>& X, Real (&v)[Dim]) {
      const Real x = X[0], y = X[1], z = X[2];
      v[0] = y; v[1] = z; v[2] = x;
   };

   // --------------------------
   // Kernel policy
   // --------------------------
#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<num_quad_1d, num_quad_1d, num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   // --------------------------
   // Build the FACE operator and do one application
   // --------------------------
   // Factory chooses BC=Empty for now.
   auto face_op = MakeAdvectionFaceOperator<KernelPolicy>(fe_space, face_mesh, int_rules, adv);

   const Integer ndofs = fe_space.GetNumberOfFiniteElementDofs();
   Vector dofs_in(ndofs), dofs_out(ndofs);
   dofs_in = 1.0;
   face_op(dofs_in, dofs_out); // compile/link sanity check

   return 0;
}
