// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace gendil;

int main(int argc, char *argv[])
{   
   const Integer num_elem_1d = 5;
   /////////
   // mesh 1
   const Real h_1 = Real(1.0);
   const Integer n1 = num_elem_1d;
   const Integer n2 = num_elem_1d;
   const Integer n3 = num_elem_1d;
   Cartesian3DMesh mesh_1( h_1, n1, n2, n3 );

   /////////
   // mesh 2
   const Real h_2 = Real(2.0);
   const Integer n4 = num_elem_1d;
   const Integer n5 = num_elem_1d;
   Cartesian2DMesh mesh_2( h_2, n4, n5 );

   /////////
   // mesh 3
   const Real h_3 = Real(1.0);
   const Integer n6 = num_elem_1d;
   Cartesian1DMesh mesh_3( h_3, n6 );

   //////////////////////
   // Tensor product Mesh
   auto mesh = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

   ///////////////////////
   // Finite Element Space
   constexpr Integer order = 2;

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
   constexpr Integer Dim = GetDim( fe_space );

   ///////////////////
   // Integration Rule
   constexpr Integer num_quad = order + 1;

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

   auto sigma = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim > const & X ) -> Real
   {
      return Real(1.0);
   };

#if defined(GENDIL_USE_DEVICE)
   // using ThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad>;
   using ThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad,num_quad>;
   // using ThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad,num_quad,num_quad>;
   constexpr size_t NumSharedDimensions = Dim;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto sol_operator = MakeSpeedOfLightOperator< KernelPolicy >( fe_space, int_rules );
   auto mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( fe_space, int_rules, sigma );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   Vector dofs_in( num_dofs );
   Vector dofs_out( num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   std::cout << "Order:" << order << "\n";
   std::cout << "Num Quads:" << num_quad << "\n";
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Number of dofs:" << num_dofs << "\n";

   dofs_in = Real(1.0);
   sol_operator( dofs_in, dofs_out );

   GENDIL_DEVICE_SYNC;
   const Integer num_iter = 5;
   const auto sol_start = std::chrono::steady_clock::now();
   for ( Integer iter = 0; iter < num_iter; iter++ )
   {
      sol_operator( dofs_out, dofs_in );
      sol_operator( dofs_in, dofs_out );
   }
   GENDIL_DEVICE_SYNC;
   const auto sol_end = std::chrono::steady_clock::now();

   std::cout << "\n gendil Speed-of-light Mass 6D\n";
   const std::chrono::duration<double> sol_elapsed_seconds = sol_end - sol_start;
   std::cout << "Total time:" << sol_elapsed_seconds.count() << "s\n";
   const Integer total_iter = 2 * num_iter;
   const double sol_time_per_iter = sol_elapsed_seconds.count() / total_iter;
   std::cout << "Number of iterations:" << total_iter << "\n";
   std::cout << "Time per iteration:" << sol_time_per_iter << "s\n";
   std::cout << "Throughput:" << num_dofs / sol_time_per_iter << "Dofs/s\n";

   GENDIL_DEVICE_SYNC;
   const auto start = std::chrono::steady_clock::now();
   for ( Integer iter = 0; iter < num_iter; iter++ )
   {
      mass_operator( dofs_out, dofs_in );
      mass_operator( dofs_in, dofs_out );
   }
   GENDIL_DEVICE_SYNC;
   const auto end = std::chrono::steady_clock::now();

   std::cout << "\n gendil Mass 6D\n";
   const std::chrono::duration<double> elapsed_seconds = end - start;
   std::cout << "Total time:" << elapsed_seconds.count() << "s\n";
   const double time_per_iter = elapsed_seconds.count() / total_iter;
   std::cout << "Number of iterations:" << total_iter << "\n";
   std::cout << "Time per iteration:" << time_per_iter << "s\n";
   std::cout << "Throughput:" << num_dofs / time_per_iter << "Dofs/s\n";

   // dofs_out.Print( mfem::out, num_elem_dofs );

   return 0;
}
