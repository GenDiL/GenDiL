// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace mfem;
using namespace gendil;

int main(int argc, char ** argv)
{
   const Integer num_elem_1d = 3;
   const Real h = 1.0/num_elem_1d;

   /////////
   // mesh 1
   const Real h_1 = h;
   const Integer nx_1 = num_elem_1d;
   Cartesian1DMesh mesh_1( h_1, nx_1 );

   /////////
   // mesh 2
   const Real h_2 = h;
   const Integer nx_2 = num_elem_1d;
   Cartesian1DMesh mesh_2( h_2, nx_2 );

   /////////
   // mesh 3
   const Real h_3 = h;
   const Integer n_3 = num_elem_1d;
   Cartesian1DMesh mesh_3( h_3, n_3 );

   //////////////////////
   // Tensor product Mesh
   auto mesh = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

   ///////////////////////
   // Finite Element Space

   constexpr Integer order = 2;

   // Orders
   FiniteElementOrders< order, order, order > orders;

   // Finite element
   auto finite_element = MakeLobattoFiniteElement( orders );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   constexpr Integer Dim = 3;

   ///////////////////
   // Integration Rule

   constexpr Integer num_quad_1d = order+2;

   // Number of quadrature points
   IntegrationRuleNumPoints< num_quad_1d, num_quad_1d, num_quad_1d > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   auto velocity = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return x*y*z;
   };

#if defined(GENDIL_USE_DEVICE)
   #if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
   #elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
   #endif
   constexpr Integer NumSharedDimensions = 3;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   const char device_config[] = "cpu";
   using KernelPolicy = SerialKernelConfiguration;
#endif
   std::cout << "Config: " << device_config << std::endl;
   mfem::Device device(device_config);
   device.Print();

   const double sigma = 1.0;
   const double kappa = (order+1)*(order+1);
   auto diffusion_operator = MakeDiffusionOperator< KernelPolicy >( fe_space, int_rules, velocity, sigma, kappa );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector dofs_in( num_dofs );
   mfem::Vector dofs_out_mf( num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";

   dofs_in.Randomize(1);
   // dofs_in = 0.0;
   // dofs_in.HostWrite();
   // dofs_in[0] = 1.0;

   diffusion_operator.Mult( dofs_in, dofs_out_mf );

   /// Using MFEM partial assembly DG
   auto non_periodic_mesh = mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON, 1.0, 1.0, 1.0, false );

   // Create translation vectors defining the periodicity
   mfem::Vector x_translation({1.0, 0.0, 0.0});
   mfem::Vector y_translation({0.0, 1.0, 0.0});
   mfem::Vector z_translation({0.0, 0.0, 1.0});
   std::vector<mfem::Vector> translations = {x_translation, y_translation, z_translation};
   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   // Mesh periodic_mesh = Mesh::MakePeriodic(non_periodic_mesh, non_periodic_mesh.CreatePeriodicVertexMapping(translations));
   int dim = non_periodic_mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&non_periodic_mesh, &fec);

   GridFunction dofs_out_pa( &fes );

   auto velocity_mfem = [=]( const mfem::Vector & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return x*y*z;
   };

   FunctionCoefficient velocity_c(velocity_mfem);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator( new DiffusionIntegrator(velocity_c) );
   blf_pa.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(velocity_c, sigma, kappa));
   blf_pa.AddBdrFaceIntegrator(new DGDiffusionIntegrator(velocity_c, sigma, kappa));
   blf_pa.Assemble();
   blf_pa.Mult( dofs_in, dofs_out_pa );

   std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;
   std::cout << "\nmf vector:\n";
   dofs_out_mf.Print( std::cout, num_elem_dofs );
   std::cout << "\npa vector:\n";
   dofs_out_pa.Print( std::cout, num_elem_dofs );

   std::cout << "min(Apsi_mf), max(Apsi_mf): " << dofs_out_mf.Min() << ", " << dofs_out_mf.Max() << endl;
   std::cout << "min(Apsi_pa), max(Apsi_pa): " << dofs_out_pa.Min() << ", " << dofs_out_pa.Max() << endl;
   
   dofs_out_mf -= dofs_out_pa;

   const Real absolute_error = dofs_out_mf.Norml2();
   const Real relative_error = dofs_out_mf.Norml2()/dofs_out_pa.Norml2();
   std::cout << "Absolute norm of the diff: " << absolute_error << std::endl;
   std::cout << "Relative norm of the diff: " << relative_error << std::endl;

   const Real tolerance = 1e-6;
   if( relative_error > tolerance )
   {
      return 1;
   }

   return 0;
}
