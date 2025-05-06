// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace mfem;
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

   const Real Lz = 1.0;
   const Integer num_elem_1d = 3;
   const Real h = Lz/num_elem_1d;

   /////////
   // mesh 1
   const Real h_1 = h;
   const Integer n1 = num_elem_1d;
   Cartesian1DMesh mesh_1( h_1, n1 );

   /////////
   // mesh 2
   const Real h_2 = h;
   const Integer n2 = num_elem_1d;
   Cartesian1DMesh mesh_2( h_2, n2 );

   /////////
   // mesh 3
   const Real h_3 = h;
   const Integer n3 = num_elem_1d;
   Cartesian1DMesh mesh_3( h_3, n3 );

   /////////////////////////
   // Cartesian Product Mesh
   auto mesh = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

   ///////////////////////
   // Finite Element Space

   constexpr Integer order = 1;

   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders;

   // Finite element
   auto finite_element = MakeLobattoFiniteElement( orders );

   // MFEM Finite element space
   auto mfem_mesh = mfem::Mesh::MakeCartesian3D( n1, n2, n3, mfem::Element::Type::HEXAHEDRON, h*n1, h*n2, h*n3, false );
   int dim = mfem_mesh.Dimension();
   H1_FECollection fec(order, dim);
   // L2_FECollection fec(order, dim);
   mfem::FiniteElementSpace fes(&mfem_mesh, &fec);

   // GenDiL Finite element space
   auto restriction = GetH1Restriction( fes );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element, restriction );
   // auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   ///////////////////
   // Integration Rule

   constexpr Integer num_quad_1d = order+2;

   // Space number of quadrature points
   constexpr Integer num_quad_X = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_X > num_quad_space;

   // Angle number of quadrature points
   constexpr Integer num_quad_polar = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_polar > num_quad_angle;

   // Energy number of quadrature points
   constexpr Integer num_quad_ener = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_ener > num_quad_energy;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quad_space, num_quad_angle, num_quad_energy );

   constexpr Integer Dim = GetDim( fe_space );
   auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return x*y*z;
   };

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 2;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( fe_space, int_rules, sigma );

   const mfem::IntegrationRule *ir = nullptr;
   
   GridFunction dofs_out_pa( &fes );

   auto sigma_func = [=](const Vector& X)
   {
      const Real x = X(0);
      const Real y = X(1);
      const Real z = X(2);
      return x*y*z;
   };
   FunctionCoefficient sigma_cf(sigma_func);

   // MFEM operator
   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new MassIntegrator(sigma_cf,ir));
   blf_pa.Assemble();

   FiniteElementVector dofs_in( fe_space );
   FiniteElementVector dofs_out_mf( fe_space );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   const Integer num_dofs = num_elem * num_elem_dofs;
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   dofs_in.Randomize(1);
   // dofs_in = 0.0;
   // dofs_in[0] = 1.0;
   mass_operator.Mult( dofs_in, dofs_out_mf );

   blf_pa.Mult(dofs_in, dofs_out_pa);

   std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;
   std::cout << "in vector: ";
   dofs_in.Print( std::cout );
   std::cout << "\ngendil vector:\n";
   dofs_out_mf.Print( std::cout );
   std::cout << "\nmfem vector:\n";
   dofs_out_pa.Print( std::cout );

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
