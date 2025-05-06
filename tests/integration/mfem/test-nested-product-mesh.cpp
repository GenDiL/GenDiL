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
   const Integer num_elem_1d = 5;
   /////////
   // Mesh 1
   const Real h_1 = 1.0;
   const Integer n1 = num_elem_1d;
   const Integer n2 = num_elem_1d;
   const Integer n3 = num_elem_1d;
   Cartesian2DMesh space_mesh_xy( h_1, n1, n2 );
   Cartesian1DMesh space_mesh_z( h_1, n3 );
   auto mesh_3d = MakeCartesianProductMesh( space_mesh_xy, space_mesh_z );

   /////////
   // Mesh 2
   const Real h_2 = 1.2345;
   const Integer n4 = num_elem_1d;
   const Integer n5 = num_elem_1d;
   Cartesian2DMesh mesh_2d( h_2, n4, n5 );

   /////////
   // Mesh 3
   const Real h_3 = 1.0;
   const Integer n6 = num_elem_1d;
   Cartesian1DMesh mesh_3( h_3, n6 );

   //////////////////////
   // Tensor product Mesh
   auto mesh = MakeCartesianProductMesh( mesh_3d, mesh_2d, mesh_3 );

   ///////////////////////
   // Finite Element Space
   constexpr Integer order = 1;

   // Orders
   constexpr Integer order_1 = order;
   constexpr Integer order_2 = order;
   constexpr Integer order_3 = order;
   constexpr Integer order_4 = order;
   constexpr Integer order_5 = order;
   constexpr Integer order_6 = order;
   FiniteElementOrders< order_1, order_2, order_3, order_4, order_5, order_6 > orders;

   // Finite element
   auto finite_element = MakeLegendreFiniteElement( orders );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   ///////////////////
   // Integration Rule
   constexpr Integer num_quad = order + 2;

   // Space number of quadrature points
   constexpr Integer num_quad_1 = num_quad;
   constexpr Integer num_quad_2 = num_quad;
   constexpr Integer num_quad_3 = num_quad;
   constexpr Integer num_quad_4 = num_quad;
   constexpr Integer num_quad_5 = num_quad;
   constexpr Integer num_quad_6 = num_quad;
   IntegrationRuleNumPoints< num_quad_1, num_quad_2, num_quad_3, num_quad_4, num_quad_5, num_quad_6 > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto advection_field = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X, Real (&v)[Dim] )
   {
      v[0] = 1.0;
      v[1] = 1.0;
      v[2] = 1.0;
      v[3] = 1.0;
      v[4] = 1.0;
      v[5] = 1.0;
   };

   // Kernel configuration
   #if defined(GENDIL_USE_DEVICE)
   #if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
   #elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
   #endif
   // using ThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad>;
   using ThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad,num_quad>;
   // using ThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad,num_quad,num_quad>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   const char device_config[] = "cpu";
   using KernelPolicy = SerialKernelConfiguration;
#endif
   mfem::Device device(device_config);
   device.Print();

   auto sol_operator = MakeFaceSpeedOfLightOperator< KernelPolicy >( fe_space, int_rules );
   auto adv_operator = MakeAdvectionOperator< KernelPolicy >( fe_space, int_rules, advection_field );

   FiniteElementVector dofs_in( fe_space );
   FiniteElementVector dofs_out( fe_space );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   const Integer num_dofs = num_elem * num_elem_dofs;
   std::cout << "Order:" << order << "\n";
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Number of dofs:" << num_dofs << "\n";

   dofs_in = 1.0;
   sol_operator.Mult( dofs_in, dofs_out );

   const Integer num_iter = 1;
   const auto sol_start = std::chrono::steady_clock::now();
   for ( Integer iter = 0; iter < num_iter; iter++ )
   {
      sol_operator.Mult( dofs_out, dofs_in );
      sol_operator.Mult( dofs_in, dofs_out );
   }
   const auto sol_end = std::chrono::steady_clock::now();

   std::cout << "\n gendil Speed-of-light Advection 6D\n";
   const std::chrono::duration<double> sol_elapsed_seconds = sol_end - sol_start;
   std::cout << "Total time:" << sol_elapsed_seconds.count() << "s\n";
   const Integer total_iter = 2 * num_iter;
   const double sol_time_per_iter = sol_elapsed_seconds.count() / total_iter;
   std::cout << "Number of iterations:" << total_iter << "\n";
   std::cout << "Time per iteration:" << sol_time_per_iter << "s\n";
   std::cout << "Throughput:" << num_dofs / sol_time_per_iter << "Dofs/s\n";

   const auto start = std::chrono::steady_clock::now();
   for ( Integer iter = 0; iter < num_iter; iter++ )
   {
      adv_operator.Mult( dofs_out, dofs_in );
      adv_operator.Mult( dofs_in, dofs_out );
   }
   const auto end = std::chrono::steady_clock::now();

   std::cout << "\n gendil Advection 6D\n";
   const std::chrono::duration<double> elapsed_seconds = end - start;
   std::cout << "Total time:" << elapsed_seconds.count() << "s\n";
   const double time_per_iter = elapsed_seconds.count() / total_iter;
   std::cout << "Number of iterations:" << total_iter << "\n";
   std::cout << "Time per iteration:" << time_per_iter << "s\n";
   std::cout << "Throughput:" << num_dofs / time_per_iter << "Dofs/s\n";

   // dofs_out.Print( mfem::out, num_elem_dofs );

   return 0;
}
