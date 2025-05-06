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

   const Integer num_elem_1d = 3;
   /////////
   // Mesh 1
   const Real h_1 = 1.0;
   const Integer n1 = num_elem_1d;
   const Integer n2 = num_elem_1d;
   const Integer n3 = num_elem_1d;
   Cartesian3DMesh mesh_1( h_1, n1, n2, n3 );

   constexpr Integer mesh_order = 1;
   // auto non_periodic_mesh = mfem::Mesh::MakeCartesian3D( n1, n2, n3, mfem::Element::Type::HEXAHEDRON, h_1, h_1, h_1, false );
   // HexMesh< mesh_order > mesh_1 = MakeHexMesh< mesh_order >( non_periodic_mesh );
   std::cout << "Num cells mesh 1: "<< mesh_1.GetNumberOfCells() << std::endl;

   /////////
   // Mesh 2
   const Real h_2 = 2.0;
   const Integer n4 = num_elem_1d;
   const Integer n5 = num_elem_1d;
   // Cartesian1DMesh mesh_2( h_2, n4, n5 );
   // Cartesian2DMesh mesh_2( h_2, n4, n5 );

   auto mfem_mesh_2 = mfem::Mesh::MakeCartesian2D( n4, n5, mfem::Element::Type::QUADRILATERAL, false, h_2, h_2, false );
   QuadMesh< mesh_order > mesh_2 = MakeQuadMesh< mesh_order >( mfem_mesh_2 );
   std::cout << "Num cells mesh 2: "<< mesh_2.GetNumberOfCells() << std::endl;

   /////////
   // Mesh 3
   const Real h_3 = 1.0;
   const Integer n6 = num_elem_1d;
   // Cartesian1DMesh mesh_3( h_3, n6 );

   auto mfem_mesh_3 = mfem::Mesh::MakeCartesian1D( n6, h_3 );
   LineMesh< mesh_order > mesh_3 = MakeLineMesh< mesh_order >( mfem_mesh_3 );
   std::cout << "Num cells mesh 3: "<< mesh_3.GetNumberOfCells() << std::endl;

   //////////////////////
   // Tensor product Mesh
   auto mesh_6D = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

   ///////////////////////
   // Finite Element Space
   constexpr Integer order = 3;

   // Mesh 1 orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders_mesh_1;
   
   // Mesh 2 orders
   constexpr Integer order_U = order;
   constexpr Integer order_V = order;
   FiniteElementOrders< order_U, order_V > orders_mesh_2;

   // Mesh 3 order
   constexpr Integer order_A = order;
   FiniteElementOrders< order_A > orders_mesh_3;

   // Finite element
   auto space_finite_element = MakeLegendreFiniteElement( orders_mesh_1 );
   auto phase_finite_element = MakeLegendreFiniteElement( orders_mesh_1, orders_mesh_2, orders_mesh_3 );

   // Finite element space
   auto fe_space_3D = MakeFiniteElementSpace( mesh_1, space_finite_element );
   auto fe_space_6D = MakeFiniteElementSpace( mesh_6D, phase_finite_element );

   ///////////////////
   // Integration Rule
   constexpr Integer num_quad = order + 1;

   // Mesh 1 number of quadrature points
   constexpr Integer num_quad_X = num_quad;
   constexpr Integer num_quad_Y = num_quad;
   constexpr Integer num_quad_Z = num_quad;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quad_1;

   // Mesh 2 number of quadrature points
   constexpr Integer num_quad_U = num_quad + 1;
   constexpr Integer num_quad_V = num_quad + 1;
   IntegrationRuleNumPoints< num_quad_U, num_quad_V > num_quad_2;

   // Mesh 3 number of quadrature points
   constexpr Integer num_quad_A = num_quad + 1;
   IntegrationRuleNumPoints< num_quad_A > num_quad_3;

   // High-dimension integration rule
   auto int_rule_6D = MakeIntegrationRule( num_quad_1, num_quad_2, num_quad_3 );
   auto int_rule_3D = MakeIntegrationRule( num_quad_1 );

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   constexpr Integer TrialDim = GetDim( fe_space_6D );
   constexpr Integer TestDim = GetDim( fe_space_3D );
   using TrialThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad>;
   // using TrialThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad,num_quad>;
   // using TrialThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad,num_quad,num_quad,num_quad>;
   using TestThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad>;
   constexpr size_t TrialNumSharedDimensions = TrialDim-1;
   constexpr size_t TestNumSharedDimensions = TestDim;
   using TrialKernelPolicy = ThreadFirstKernelConfiguration< TrialThreadLayout, TrialNumSharedDimensions >;
   using TestKernelPolicy = ThreadFirstKernelConfiguration< TestThreadLayout, TestNumSharedDimensions >;
#else
   using TrialKernelPolicy = SerialKernelConfiguration;
   using TestKernelPolicy = SerialKernelConfiguration;
#endif

   auto L2_projection_operator = MakeL2Projection< TrialKernelPolicy, TestKernelPolicy >( fe_space_6D, fe_space_3D, int_rule_6D, int_rule_3D );
   // auto L2_projection_operator = MakeLinfProjection< TrialKernelPolicy >( fe_space_6D, fe_space_3D, int_rule_6D, int_rule_3D );
   auto Linf_projection_operator = MakeLinfProjection< TrialKernelPolicy >( fe_space_6D, fe_space_3D, int_rule_6D, int_rule_3D );

   FiniteElementVector dofs_in( fe_space_6D );
   FiniteElementVector dofs_out( fe_space_3D );

   const Integer num_elem_dofs = phase_finite_element.GetNumDofs();
   const Integer num_elem = fe_space_6D.GetNumberOfFiniteElements();
   const Integer num_dofs = num_elem * num_elem_dofs;
   std::cout << "Order:" << order << "\n";
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Number of dofs:" << num_dofs << "\n";

   dofs_in = 1.0;

   const Integer num_iter = 1;
   const Integer total_iter = num_iter;

   {
      const auto start = std::chrono::steady_clock::now();
      L2_projection_operator.Mult( dofs_in, dofs_out );
      MFEM_DEVICE_SYNC;
      const auto end = std::chrono::steady_clock::now();

      std::cout << "\n L2 projection 6D\n";
      const std::chrono::duration<double> elapsed_seconds = end - start;
      std::cout << "Total time:" << elapsed_seconds.count() << "s\n";
      const double time_per_iter = elapsed_seconds.count() / total_iter;
      std::cout << "Number of iterations:" << total_iter << "\n";
      std::cout << "Time per iteration:" << time_per_iter << "s\n";
      std::cout << "Throughput:" << num_dofs / time_per_iter << "Dofs/s\n";
   }

   {
      const auto start = std::chrono::steady_clock::now();
      Linf_projection_operator.Mult( dofs_in, dofs_out );
      MFEM_DEVICE_SYNC;
      const auto end = std::chrono::steady_clock::now();

      std::cout << "\n Linf projection 6D\n";
      const std::chrono::duration<double> elapsed_seconds = end - start;
      std::cout << "Total time:" << elapsed_seconds.count() << "s\n";
      const double time_per_iter = elapsed_seconds.count() / total_iter;
      std::cout << "Number of iterations:" << total_iter << "\n";
      std::cout << "Time per iteration:" << time_per_iter << "s\n";
      std::cout << "Throughput:" << num_dofs / time_per_iter << "Dofs/s\n";
   }

   // dofs_out.Print( mfem::out, num_elem_dofs );

   return 0;
}
