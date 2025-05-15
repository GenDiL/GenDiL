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

   constexpr Integer order = 4;

   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders;

   // Finite element
   auto finite_element = MakeLobattoFiniteElement( orders );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   ///////////////////
   // Integration Rule

   constexpr Integer num_quad_1d = order+2;

   // Number of quadrature points
   constexpr Integer num_quad_X = num_quad_1d;
   constexpr Integer num_quad_Y = num_quad_1d;
   constexpr Integer num_quad_Z = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return 1.0+x*y*z;
   };

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 2;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto mass_inverse_operator = MakeMassInverseFiniteElementOperator< KernelPolicy >( fe_space, int_rules, sigma );
   auto mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( fe_space, int_rules, sigma );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector rhs( num_dofs );
   mfem::Vector solution_gendil_local( num_dofs );
   mfem::Vector solution_gendil_global( num_dofs );
   mfem::Vector solution_mfem( num_dofs );
   mfem::Vector diff( num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   rhs.Randomize(1);
   std::cout << "GenDiL solve using local CG.\n";
   // rhs = 1.0;
   mass_inverse_operator.Mult( rhs, solution_gendil_local );

   const int cg_print_iter = 2;
   const int cg_max_iter = 1000;
   const Real cg_rtol = 1e-14;
   const Real cg_atol = 0.0;

   solution_gendil_global = 0.0;
   std::cout << "GenDiL solve using global CG.\n";
   CG( mass_operator, rhs, solution_gendil_global, cg_print_iter, cg_max_iter, cg_rtol, cg_atol );

   std::cout << "Norm of solution_gendil: " << solution_gendil_local.Norml2() << std::endl;
   std::cout << "Norm of solution_mfem: " << solution_gendil_global.Norml2() << std::endl;
   // std::cout << "in vector: ";
   // rhs.Print( std::cout );
   // std::cout << "\nsolution_gendil:\n";
   // solution_gendil.Print( std::cout );
   // std::cout << "\nsolution_mfem:\n";
   // solution_mfem.Print( std::cout );

   diff = solution_gendil_local;
   diff -= solution_gendil_global;

   std::cout << "Absolute norm of the diff (local/global): " << diff.Norml2() << std::endl;
   std::cout << "Relative norm of the diff (local/global): " << diff.Norml2()/solution_gendil_global.Norml2() << std::endl;

   // MFEM Legacy operator
   auto mfem_mesh = mfem::Mesh::MakeCartesian3D( n1, n2, n3, mfem::Element::Type::HEXAHEDRON, h*n1, h*n2, h*n3, false );
   int dim = mfem_mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&mfem_mesh, &fec);

   const mfem::IntegrationRule *ir = nullptr;
   
   GridFunction dofs_out_pa( &fes );

   auto sigma_func = [=](const mfem::Vector& X)
   {
      const Real x = X(0);
      const Real y = X(1);
      const Real z = X(2);
      return 1.0+x*y*z;
   };
   FunctionCoefficient sigma_cf(sigma_func);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new MassIntegrator(sigma_cf,ir));
   blf_pa.Assemble();

   std::cout << "MFEM solve using global CG.\n";
   solution_mfem = 0.0;
   CG( blf_pa, rhs, solution_mfem, cg_print_iter, cg_max_iter, cg_rtol, cg_atol );
   std::cout << "Norm of solution_mfem: " << solution_mfem.Norml2() << std::endl;
   // std::cout << "in vector: ";
   // rhs.Print( std::cout );
   // std::cout << "\nglobal CG solution:\n";
   // solution_gendil_global.Print( std::cout );
   // std::cout << "\nlocal CG solution:\n";
   // solution_gendil_local.Print( std::cout );

   diff = solution_gendil_local;
   diff -= solution_mfem;

   const Real tolerance = 1e-6;

   const Real local_absolute_error = diff.Norml2();
   const Real local_relative_error = diff.Norml2()/solution_mfem.Norml2();
   std::cout << "Absolute norm of the diff (gendil-local/mfem): " << local_absolute_error << std::endl;
   std::cout << "Relative norm of the diff (gendil-local/mfem): " << local_relative_error << std::endl;
   
   diff = solution_gendil_global;
   diff -= solution_mfem;

   const Real global_absolute_error = diff.Norml2();
   const Real global_relative_error = diff.Norml2()/solution_mfem.Norml2();
   std::cout << "Absolute norm of the diff (gendil-global/mfem): " << global_absolute_error << std::endl;
   std::cout << "Relative norm of the diff (gendil-global/mfem): " << global_relative_error << std::endl;

   if( local_relative_error > tolerance )
   {
      return 1;
   }

   if( global_relative_error > tolerance )
   {
      return 1;
   }

   return 0;
}
