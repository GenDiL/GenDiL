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

   const Integer num_elem_1d = 3; // !FIXME use 3
   const Real h = 1.0/num_elem_1d;

   /////////
   // Mesh 1
   const Real h_space = h;
   const Integer nx = num_elem_1d;
   PeriodicCartesian1DMesh mesh_x( h_space, nx );
   // Cartesian1DMesh mesh_x( h_space, nx );

   /////////
   // Mesh 2
   const Real h_angle = h;
   const Integer ny = num_elem_1d;
   PeriodicCartesian1DMesh mesh_y( h_angle, ny );
   // Cartesian1DMesh mesh_y( h_angle, ny );

   /////////
   // Mesh 3
   const Real h_energy = h;
   const Integer nz = num_elem_1d;
   PeriodicCartesian1DMesh mesh_z( h_energy, nz );
   // Cartesian1DMesh mesh_z( h_energy, nz );

   //////////////////////
   // Tensor product Mesh
   // auto mesh = MakeCartesianProductMesh( mesh_x, mesh_y, mesh_z );
   PeriodicCartesian3DMesh mesh( h, h, h, nx, ny, nz, { 0.0, 0.0, 0.0 } );

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
   auto adv = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim> & X, Real (&v)[Dim] )
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      v[0] = y;
      v[1] = z;
      v[2] = x;
   };
   auto zero = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X ){ return 0.0; };

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto advection_operator = MakeAdvectionOperator< KernelPolicy >( fe_space, int_rules, adv, zero );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector dofs_in( num_dofs );
   mfem::Vector dofs_out_mf( num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   dofs_in.Randomize(1);
   // dofs_in.HostWrite();
   // dofs_in = 0.0;
   // dofs_in[0] = 1.0;
   // dofs_in = 1.0;
   advection_operator.Mult( dofs_in, dofs_out_mf );

   /// Using MFEM partial assembly DG
   auto non_periodic_mesh = mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON, 1.0, 1.0, 1.0, false );
   // Create translation vectors defining the periodicity
   mfem::Vector x_translation({1.0, 0.0, 0.0});
   mfem::Vector y_translation({0.0, 1.0, 0.0});
   mfem::Vector z_translation({0.0, 0.0, 1.0});
   std::vector<mfem::Vector> translations = {x_translation, y_translation, z_translation};
   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   Mesh periodic_mesh = Mesh::MakePeriodic(non_periodic_mesh, non_periodic_mesh.CreatePeriodicVertexMapping(translations));
   int dim = non_periodic_mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&periodic_mesh, &fec);
   
   GridFunction dofs_out_pa( &fes );

   auto adv_func = [=](const mfem::Vector& x, mfem::Vector& v)
   {
      const Real X = x(0);
      const Real Y = x(1);
      const Real Z = x(2);
      v[0] = Y;
      v[1] = Z;
      v[2] = X;
   };
   VectorFunctionCoefficient velocity(3, adv_func);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   constexpr double alpha = 1.0; // Should discretize div(adv*psi)
   blf_pa.AddDomainIntegrator(
      new ConservativeConvectionIntegrator(velocity, alpha));
   blf_pa.AddInteriorFaceIntegrator(
      new ConservativeDGTraceIntegrator(velocity, alpha));
   blf_pa.AddBdrFaceIntegrator(
      new ConservativeDGTraceIntegrator(velocity, alpha));
   blf_pa.Assemble();
   blf_pa.Mult(dofs_in, dofs_out_pa);

   std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;

   std::cout << "min(Apsi_mf), max(Apsi_mf): " << dofs_out_mf.Min() << ", " << dofs_out_mf.Max() << endl;
   std::cout << "min(Apsi_pa), max(Apsi_pa): " << dofs_out_pa.Min() << ", " << dofs_out_pa.Max() << endl;
   // std::cout << "in vector: ";
   // dofs_in.Print( std::cout, num_elem_dofs );
   // std::cout << "\nmf vector:\n";
   // dofs_out_mf.Print( std::cout, num_elem_dofs );
   // std::cout << "\npa vector:\n";
   // dofs_out_pa.Print( std::cout, num_elem_dofs );
   // for( size_t i=0; i < dofs_out_mf.Size(); i++)
   // {
   //    const Real diff = dofs_out_mf[i] - dofs_out_pa[i];
   //    const Real distance = diff * diff;
   //    if( distance > 1e-10 )
   //    {
   //       std::cout << i << std::endl;
   //    }
   // }

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
