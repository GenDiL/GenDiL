// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>
#include <gendil/gendil.hpp>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;
using namespace gendil;

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

   constexpr int num_elem_1d = 8;
   constexpr int mesh_order = 2;
   const int ref_levels = 0;

   mfem::Mesh mfem_mesh = mfem::Mesh::MakeCartesian1D(num_elem_1d);
   mfem_mesh.SetCurvature(mesh_order);

   for (int lev = 0; lev < ref_levels; ++lev)
   {
      mfem_mesh.UniformRefinement();
   }

   // Convert to GenDiL mesh
   LineMesh<mesh_order> mesh = MakeLineMesh<mesh_order>(mfem_mesh);

   constexpr Integer order = 2;
   FiniteElementOrders<order> orders;
   auto finite_element = MakeLobattoFiniteElement(orders);
   auto fe_space = MakeFiniteElementSpace(mesh, finite_element);

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints<num_quad_1d> num_quads;
   auto int_rules = MakeIntegrationRule(num_quads);

   constexpr Integer Dim = GetDim(fe_space);
   auto adv = [=] GENDIL_HOST_DEVICE(const std::array<Real, Dim> &X, Real (&v)[Dim])
   {
      v[0] = 1.2345;
   };
   auto zero = [] GENDIL_HOST_DEVICE(const std::array<Real, Dim> &X) { return 0.0; };

#if defined(GENDIL_USE_DEVICE)
   using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration<ThreadLayout, 1>;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto advection_operator = MakeAdvectionOperator<KernelPolicy>(fe_space, int_rules, adv, zero);

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector dofs_in(num_dofs), dofs_out_mf(num_dofs);

   dofs_in.Randomize(1);
   advection_operator.Mult(dofs_in, dofs_out_mf);

   // MFEM side
   const int dim = mfem_mesh.Dimension();
   mfem::L2_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&mfem_mesh, &fec);
   mfem::GridFunction dofs_out_pa(&fes);

   auto adv_func = [](const mfem::Vector &x, mfem::Vector &v)
   {
      v.SetSize(1);
      v[0] = 1.2345;
   };
   mfem::VectorFunctionCoefficient velocity(1, adv_func);

   mfem::BilinearForm blf_pa(&fes);
//    blf_pa.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL); // NOT SUPPORTED
   constexpr double alpha = 1.0;
   blf_pa.AddDomainIntegrator(new mfem::ConservativeConvectionIntegrator(velocity, alpha));
   blf_pa.AddInteriorFaceIntegrator(new mfem::ConservativeDGTraceIntegrator(velocity, alpha));
   blf_pa.AddBdrFaceIntegrator(new mfem::ConservativeDGTraceIntegrator(velocity, alpha));
   blf_pa.Assemble();
   blf_pa.Mult(dofs_in, dofs_out_pa);

   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;

   dofs_out_mf -= dofs_out_pa;
   const Real absolute_error = dofs_out_mf.Norml2();
   const Real relative_error = absolute_error / dofs_out_pa.Norml2();
   std::cout << "Absolute norm of the diff: " << absolute_error << std::endl;
   std::cout << "Relative norm of the diff: " << relative_error << std::endl;

   const Real tolerance = 1e-6;
   if (relative_error > tolerance)
   {
      std::cerr << "Test failed with relative error " << relative_error << std::endl;
      return 1;
   }

   std::cout << "Test passed!" << std::endl;
   return 0;
}
