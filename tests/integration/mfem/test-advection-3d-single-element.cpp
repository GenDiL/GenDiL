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

   const Integer num_elem_1d = 1;
   const Real h = 1.0/num_elem_1d;

   //////////////
   // mesh x
   const Real h_space = h;
   const Integer nx = num_elem_1d;
//    PeriodicCartesian1DMesh mesh_x( h_space, nx );
   Cartesian1DMesh mesh_x( h_space, nx );

   //////////////
   // mesh y
   const Real h_angle = h;
   const Integer ny = num_elem_1d;
//    PeriodicCartesian1DMesh mesh_y( h_angle, ny );
   Cartesian1DMesh mesh_y( h_angle, ny );

   ///////////////
   // mesh z
   const Real h_energy = h;
   const Integer nz = num_elem_1d;
//    PeriodicCartesian1DMesh mesh_z( h_energy, nz );
   Cartesian1DMesh mesh_z( h_energy, nz );

   //////////////////////
   // Tensor product Mesh
   // auto mesh = MakeCartesianProductMesh( mesh_x, mesh_y, mesh_z );
//    PeriodicCartesian3DMesh mesh( h, h, h, nx, ny, nz, { 0.0, 0.0, 0.0 } );
   Cartesian3DMesh mesh( h, h, h, nx, ny, nz, { 0.0, 0.0, 0.0 } );

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

   constexpr Integer num_quad_1d = order+2; // !FIXME 2;

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
   using ThreadLayout1d = ThreadBlockLayout<num_quad_1d>;
   using ThreadLayout2d = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   using ThreadLayout3d = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   constexpr size_t NumSharedDimensions = Dim;
#else
   using ThreadLayout1d = ThreadBlockLayout<>;
   using ThreadLayout2d = ThreadBlockLayout<>;
   using ThreadLayout3d = ThreadBlockLayout<>;
   constexpr size_t NumSharedDimensions = 0;
#endif
   using ThreadingPolicy1d = ThreadFirstKernelConfiguration< ThreadLayout1d, NumSharedDimensions >;
   using ThreadingPolicy2d = ThreadFirstKernelConfiguration< ThreadLayout2d, NumSharedDimensions >;
   using ThreadingPolicy3d = ThreadFirstKernelConfiguration< ThreadLayout3d, NumSharedDimensions >;

   auto advection_operator_1d = MakeAdvectionOperator< ThreadingPolicy1d >( fe_space, int_rules, adv, zero );
   auto advection_operator_2d = MakeAdvectionOperator< ThreadingPolicy2d >( fe_space, int_rules, adv, zero );
   auto advection_operator_3d = MakeAdvectionOperator< ThreadingPolicy3d >( fe_space, int_rules, adv, zero );

   FiniteElementVector dofs_in( fe_space );
   FiniteElementVector dofs_out_mf_1d( fe_space );
   FiniteElementVector dofs_out_mf_2d( fe_space );
   FiniteElementVector dofs_out_mf_3d( fe_space );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   const Integer num_dofs = num_elem * num_elem_dofs;
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   dofs_in.Randomize(1);
   // dofs_in.HostWrite();
   // dofs_in = 0.0;
   // dofs_in[0] = 1.0;
   // dofs_in = 1.0;
   std::cout << "\nAdvection 1D" << std::endl;
   advection_operator_1d.Mult( dofs_in, dofs_out_mf_1d );
   GENDIL_DEVICE_SYNC;
   std::cout << "\nAdvection 2D" << std::endl;
   advection_operator_2d.Mult( dofs_in, dofs_out_mf_2d );
   GENDIL_DEVICE_SYNC;
   std::cout << "\nAdvection 3D" << std::endl;
   advection_operator_3d.Mult( dofs_in, dofs_out_mf_3d );
   GENDIL_DEVICE_SYNC;

   std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   std::cout << "dofs_out_mf_1d size: " << dofs_out_mf_1d.Size() << std::endl;
   std::cout << "dofs_out_mf_2d size: " << dofs_out_mf_2d.Size() << std::endl;
   std::cout << "dofs_out_mf_3d size: " << dofs_out_mf_3d.Size() << std::endl;
   std::cout << "Norm of mf 1d: " << dofs_out_mf_1d.Norml2() << std::endl;
   std::cout << "Norm of mf 2d: " << dofs_out_mf_2d.Norml2() << std::endl;
   std::cout << "Norm of mf 3d: " << dofs_out_mf_3d.Norml2() << std::endl;

//    std::cout << "min(Apsi_mf), max(Apsi_mf): " << dofs_out_mf.Min() << ", " << dofs_out_mf.Max() << endl;
//    std::cout << "min(Apsi_pa), max(Apsi_pa): " << dofs_out_pa.Min() << ", " << dofs_out_pa.Max() << endl;
   // std::cout << "in vector: ";
   // dofs_in.Print( std::cout, num_elem_dofs );
   std::cout << "\nmf 1d vector:\n";
   dofs_out_mf_1d.Print( std::cout, num_elem_dofs );
   std::cout << "\nmf 2d vector:\n";
   dofs_out_mf_2d.Print( std::cout, num_elem_dofs );
   std::cout << "\nmf 3d vector:\n";
   dofs_out_mf_3d.Print( std::cout, num_elem_dofs );
   // for( size_t i=0; i < dofs_out_mf.Size(); i++)
   // {
   //    const Real diff = dofs_out_mf[i] - dofs_out_pa[i];
   //    const Real distance = diff * diff;
   //    if( distance > 1e-10 )
   //    {
   //       std::cout << i << std::endl;
   //    }
   // }

//    dofs_out_mf -= dofs_out_pa;

//    std::cout << "Absolute norm of the diff: " << dofs_out_mf.Norml2() << std::endl;
//    std::cout << "Relative norm of the diff: " << dofs_out_mf.Norml2()/dofs_out_pa.Norml2() << std::endl;

   return 0;
}
