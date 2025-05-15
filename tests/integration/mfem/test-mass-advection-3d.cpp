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
   const Real Lz = 1.0;
   const Integer num_elem_1d = 3;
   const Real h = Lz/num_elem_1d;

   /////////
   // Mesh 1
   const Real h_1 = h;
   const Integer n1 = num_elem_1d;
   PeriodicCartesian1DMesh mesh_1( h_1, n1 );

   /////////
   // Mesh 2
   const Real h_2 = h;
   const Integer n2 = num_elem_1d;
   PeriodicCartesian1DMesh mesh_2( h_2, n2 );

   /////////
   // Mesh 3
   const Real h_3 = h;
   const Integer n3 = num_elem_1d;
   PeriodicCartesian1DMesh mesh_3( h_3, n3 );

   //////////////////////
   // Tensor Product Mesh
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
   auto zero = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      return 0.0;
   };

   auto f = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      const Real z = X[0];
      const Real f_0 = 1.0e6;
      const Real f_1 = 1.0e-10;
      const Real f_2 = 1.0e6;
      if ( z < 0.3*Lz )
      {
         return f_0;
      }
      else if( z > 0.8*Lz )
      {
         return f_2;
      }
      else
      {
         return f_1;
      }
   };

   auto g = [] GENDIL_HOST_DEVICE ( Real x) -> Real { return 0.0; };
   auto gg = [] GENDIL_HOST_DEVICE ( Real x, Real z ) -> Real { return 0.0; };

   auto adv = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> & X, Real (&a)[Dim] )
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      a[0] = y;
      a[1] = g( x );
      a[2] = gg( x, z );
   };

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   #if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
   #elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
   #endif
   constexpr Integer NumSharedDimensions = Dim;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   const char device_config[] = "cpu";
   using KernelPolicy = SerialKernelConfiguration;
#endif
   mfem::Device device(device_config);
   device.Print();

   auto advection_operator = MakeMassAdvectionOperator< KernelPolicy >( fe_space, int_rules, adv, zero );
   auto mixed_mass_operator = MakeMixedMassOperator< KernelPolicy >( fe_space, fe_space, int_rules, f );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector dofs_in( num_dofs );
   mfem::Vector dofs_out_mf( num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   //dofs_in.Randomize(1);
   // dofs_in = 0.0;
   // dofs_in[7] = 1.0;

   dofs_in = 1.0;

   advection_operator.Mult( dofs_in, dofs_out_mf );
   const Real eps = 1e-10;
   if (fabs(dofs_out_mf.Min()) > eps || fabs(dofs_out_mf.Max()) > eps)
   {
      cout << "order: " << order << ", min(Apsi), max(Apsi): " 
           << dofs_out_mf.Min() << ", " << dofs_out_mf.Max() << " => discrete div(constant field) IS NOT zero!!!"<< endl;
   }
   else
   {
      cout << "order: " << order << ", min(Apsi), max(Apsi): " 
           << dofs_out_mf.Min() << ", " << dofs_out_mf.Max() << " => discrete div(constant field) IS zero."<< endl;
   }

   mixed_mass_operator.Mult( dofs_out_mf, dofs_in );

   // /// Using MFEM partial assembly DG
   // auto non_periodic_mesh = mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON, 1.0, 1.0, 1.0, false );
   // // Create translation vectors defining the periodicity
   // Vector x_translation({1.0, 0.0, 0.0});
   // Vector y_translation({0.0, 1.0, 0.0});
   // Vector z_translation({0.0, 0.0, 1.0});
   // std::vector<Vector> translations = {x_translation, y_translation, z_translation};
   // // Create the periodic mesh using the vertex mapping defined by the translation vectors
   // Mesh periodic_mesh = Mesh::MakePeriodic(non_periodic_mesh, non_periodic_mesh.CreatePeriodicVertexMapping(translations));
   // int dim = non_periodic_mesh.Dimension();
   // L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   // mfem::FiniteElementSpace fes(&periodic_mesh, &fec);

   // const mfem::IntegrationRule *ir = nullptr;
   
   // GridFunction dofs_out_pa( &fes );
   
   // Vector vec( dim );
   // vec = 1.0;
   // VectorConstantCoefficient velocity( vec );

   // BilinearForm blf_pa(&fes);
   // blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   // constexpr double alpha = -1.0;
   // blf_pa.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   // blf_pa.AddInteriorFaceIntegrator(new ConservativeDGTraceIntegrator(velocity, alpha));
   // blf_pa.AddBdrFaceIntegrator(new ConservativeDGTraceIntegrator(velocity, alpha));
   // blf_pa.Assemble();
   // blf_pa.Mult(dofs_in, dofs_out_pa);

   std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   // std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   // std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;
   // // std::cout << "in vector: ";
   // // dofs_in.Print( std::cout, num_elem_dofs );
   // // std::cout << "mf vector: ";
   // // dofs_out_mf.Print( std::cout, num_elem_dofs );
   // // std::cout << "pa vector: ";
   // // dofs_out_pa.Print( std::cout, num_elem_dofs );

   // dofs_out_mf -= dofs_out_pa;

   // std::cout << "Absolute norm of the diff: " << dofs_out_mf.Norml2() << std::endl;
   // std::cout << "Relative norm of the diff: " << dofs_out_mf.Norml2()/dofs_out_pa.Norml2() << std::endl;

   return 0;
}
