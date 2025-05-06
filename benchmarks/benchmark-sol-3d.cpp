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
   
   const Integer num_elem_1d = 100;

   //////////////
   // Space mesh
   const Real h_1 = 1.0;
   // const Integer n1 = num_elem_1d;
   // Cartesian1DMesh mesh_1( h_1, n1 );

   // //////////////
   // // Angle mesh
   // const Real h_2 = 2 * 3.14;
   // const Integer n2 = num_elem_1d;
   // Cartesian1DMesh mesh_2( h_2, n2 );

   // ///////////////
   // // Energy mesh
   // const Real h_3 = 1.0;
   // const Integer n3 = num_elem_1d;
   // Cartesian1DMesh mesh_3( h_3, n3 );

   // //////////////////////
   // // Tensor product Mesh
   // auto struct_mesh = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

   // Cartesian3DMesh cart3d_mesh( h_1, n1, n2, n3 );

   auto mfem_unstructured_mesh = mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON, h_1, h_1, h_1, false );
   HexMesh<1> unstruct_mesh = MakeHexMesh< 1 >( mfem_unstructured_mesh );

   ///////////////////////
   // Finite Element Space
   constexpr Integer order = 3;

   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders;

   // Finite element
   auto finite_element = MakeLegendreFiniteElement( orders );

   // Finite element space
   // auto fe_space = MakeFiniteElementSpace( struct_mesh, finite_element );
   // auto fe_space = MakeFiniteElementSpace( cart3d_mesh, finite_element );
   auto fe_space = MakeFiniteElementSpace( unstruct_mesh, finite_element );

   ///////////////////
   // Integration Rule
   constexpr Integer num_quad_1d = order + 1;

   // Number of quadrature points
   constexpr Integer num_quad_X = num_quad_1d;
   constexpr Integer num_quad_Y = num_quad_1d;
   constexpr Integer num_quad_Z = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quad;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quad );

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   constexpr Integer Dim = GetDim( fe_space );
   constexpr size_t NumSharedDimensions = Dim;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto sol_operator = MakeSpeedOfLightOperator< KernelPolicy >( fe_space, int_rules );
   auto face_sol_operator = MakeFaceSpeedOfLightOperator< KernelPolicy >( fe_space, int_rules );

   // MFEM operator
   const int dim = 3;
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   mfem::FiniteElementSpace mfem_fespace(&mfem_unstructured_mesh, &fec);
   BilinearForm mfem_operator(&mfem_fespace);
   mfem_operator.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mfem_operator.AddDomainIntegrator(new MassIntegrator);
   mfem_operator.Assemble();
   mfem_operator.Finalize();

   std::cout << "Order:" << order << "\n";
   std::cout << "Quad pts:" << num_quad_1d << "\n";
   const Integer num_iter = 5;
   double sol_throughput( 0.0 ), face_sol_throughput( 0.0 ), mfem_throughput( 0.0 );
   {
      std::cout << "\n Speed of light operator. \n";
      FiniteElementVector dofs_in( fe_space );
      FiniteElementVector dofs_out( fe_space );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = fe_space.GetNumberOfFiniteElements();
      const Integer num_dofs = num_elem * num_elem_dofs;
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
      dofs_in = 1.0;
      sol_operator.Mult( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         sol_operator.Mult( dofs_out, dofs_in );
         sol_operator.Mult( dofs_in, dofs_out );
      }
      GENDIL_DEVICE_SYNC;
      const auto end = std::chrono::steady_clock::now();

      std::cout << "Number of dofs:" << num_dofs << "\n";
      const std::chrono::duration<double> elapsed_seconds = end - start;
      std::cout << "Total time:" << elapsed_seconds.count() << "s\n";
      const Integer total_iter = 2 * num_iter;
      const double time_per_iter = elapsed_seconds.count() / total_iter;
      std::cout << "Number of iterations:" << total_iter << "\n";
      std::cout << "Time per iteration:" << time_per_iter << "s\n";
      std::cout << "Throughput:" << num_dofs / time_per_iter << "Dofs/s\n";
      sol_throughput = num_dofs / time_per_iter;
   }
   {
      std::cout << "\n Face speed of light operator. \n";
      FiniteElementVector dofs_in( fe_space );
      FiniteElementVector dofs_out( fe_space );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = fe_space.GetNumberOfFiniteElements();
      const Integer num_dofs = num_elem * num_elem_dofs;
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
      dofs_in = 1.0;
      face_sol_operator.Mult( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         face_sol_operator.Mult( dofs_out, dofs_in );
         face_sol_operator.Mult( dofs_in, dofs_out );
      }
      GENDIL_DEVICE_SYNC;
      const auto end = std::chrono::steady_clock::now();

      std::cout << "Number of dofs:" << num_dofs << "\n";
      const std::chrono::duration<double> elapsed_seconds = end - start;
      std::cout << "Total time:" << elapsed_seconds.count() << "s\n";
      const Integer total_iter = 2 * num_iter;
      const double time_per_iter = elapsed_seconds.count() / total_iter;
      std::cout << "Number of iterations:" << total_iter << "\n";
      std::cout << "Time per iteration:" << time_per_iter << "s\n";
      face_sol_throughput = num_dofs / time_per_iter;
      std::cout << "Throughput:" << face_sol_throughput << "Dofs/s\n";
   }
   {
      std::cout << "\n MFEM mass operator. \n";
      mfem::GridFunction dofs_in( &mfem_fespace );
      mfem::GridFunction dofs_out( &mfem_fespace );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = fe_space.GetNumberOfFiniteElements();
      const Integer num_dofs = num_elem * num_elem_dofs;
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
      dofs_in = 1.0;
      mfem_operator.Mult( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         mfem_operator.Mult( dofs_out, dofs_in );
         mfem_operator.Mult( dofs_in, dofs_out );
      }
      GENDIL_DEVICE_SYNC;
      const auto end = std::chrono::steady_clock::now();

      std::cout << "Number of dofs:" << num_dofs << "\n";
      const std::chrono::duration<double> elapsed_seconds = end - start;
      std::cout << "Total time:" << elapsed_seconds.count() << "s\n";
      const Integer total_iter = 2 * num_iter;
      const double time_per_iter = elapsed_seconds.count() / total_iter;
      std::cout << "Number of iterations:" << total_iter << "\n";
      std::cout << "Time per iteration:" << time_per_iter << "s\n";
      mfem_throughput = num_dofs / time_per_iter;
      std::cout << "Throughput:" << mfem_throughput << "Dofs/s\n";
   }

   std::cout << "\nSpeedup sol/mfem: " << sol_throughput / mfem_throughput << "\n";

   // dofs_out.Print( mfem::out, num_elem_dofs );

   return 0;
}
