// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifdef GENDIL_USE_MFEM
#include <mfem.hpp>
#endif

#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace gendil;

int main(int argc, char *argv[])
{
   const Integer num_elem_1d = 100;

   /////////
   // Mesh 1
   const Real h_1 = 1.0;
   const Integer n1 = num_elem_1d;
   Cartesian1DMesh mesh_1( h_1, n1 );

   /////////
   // Mesh 2
   const Real h_2 = 2.0;
   const Integer n2 = num_elem_1d;
   Cartesian1DMesh mesh_2( h_2, n2 );

   /////////
   // Mesh 3
   const Real h_3 = 1.0;
   const Integer n3 = num_elem_1d;
   Cartesian1DMesh mesh_3( h_3, n3 );

   Cartesian3DMesh cart3d_mesh( h_1, n1, n2, n3 );

   //////////////////////
   // Tensor product Mesh
   auto struct_mesh = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

#ifdef GENDIL_USE_MFEM
   auto mfem_unstructured_mesh = mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON, 1.0, 1.0, 1.0, false );
   HexMesh<1> unstruct_mesh = MakeHexMesh< 1 >( mfem_unstructured_mesh );
#endif

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
   auto struct_fe_space = MakeFiniteElementSpace( struct_mesh, finite_element );
   auto cart3d_fe_space = MakeFiniteElementSpace( cart3d_mesh, finite_element );
#ifdef GENDIL_USE_MFEM
   auto unstruct_fe_space = MakeFiniteElementSpace( unstruct_mesh, finite_element );
#endif

   constexpr Integer Dim = GetDim( struct_fe_space );

   ///////////////////
   // Integration Rule
   constexpr Integer num_quad_1d = order + 2;

   // Number of quadrature points
   constexpr Integer num_quad_X = num_quad_1d;
   constexpr Integer num_quad_Y = num_quad_1d;
   constexpr Integer num_quad_Z = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
#ifdef GENDIL_USE_MFEM
   #if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
   #elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
   #endif
#endif
   constexpr Integer NumSharedDimensions = 3;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
#ifdef GENDIL_USE_MFEM
   const char device_config[] = "cpu";
#endif
   using KernelPolicy = SerialKernelConfiguration;
#endif
#ifdef GENDIL_USE_MFEM
   mfem::Device device(device_config);
   device.Print();
#endif

   auto sol_operator = MakeSpeedOfLightOperator< KernelPolicy >( struct_fe_space, int_rules );

   auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return x*y*z;
   };

   auto struct_mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( struct_fe_space, int_rules, sigma );
   auto cart3d_mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( cart3d_fe_space, int_rules, sigma );
#ifdef GENDIL_USE_MFEM
   auto unstruct_mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( unstruct_fe_space, int_rules, sigma );

   // MFEM operator
   const int dim = 3;
   mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
   mfem::FiniteElementSpace mfem_fespace(&mfem_unstructured_mesh, &fec);
   mfem::BilinearForm mfem_operator(&mfem_fespace);
   mfem_operator.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   mfem_operator.AddDomainIntegrator(new mfem::MassIntegrator);
   mfem_operator.Assemble();
   mfem_operator.Finalize();
#endif

   std::cout << "Order:" << order << "\n";
   std::cout << "Quad pts:" << num_quad_1d << "\n";
   const Integer num_iter = 5;
   double sof_throughput( 0.0 ), struct_throughput( 0.0 ), cart3d_throughput( 0.0 );
#ifdef GENDIL_USE_MFEM
   double unstruct_throughput( 0.0 ), mfem_throughput( 0.0 );
#endif

   {
      std::cout << "\n Speed of light operator. \n";
      const Integer num_dofs = struct_fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = struct_fe_space.GetNumberOfFiniteElements();
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
      dofs_in = 1.0;
      sol_operator( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         sol_operator( dofs_out, dofs_in );
         sol_operator( dofs_in, dofs_out );
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
      sof_throughput = num_dofs / time_per_iter;
   }
   {
      std::cout << "\n Tensor mesh mass operator. \n";
      const Integer num_dofs = struct_fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = struct_fe_space.GetNumberOfFiniteElements();
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
      dofs_in = 1.0;
      struct_mass_operator( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         struct_mass_operator( dofs_out, dofs_in );
         struct_mass_operator( dofs_in, dofs_out );
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
      struct_throughput = num_dofs / time_per_iter;
      std::cout << "Throughput:" << struct_throughput << "Dofs/s\n";
   }
   {
      std::cout << "\n Cartesian 3d mesh mass operator. \n";
      const Integer num_dofs = cart3d_fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = cart3d_fe_space.GetNumberOfFiniteElements();
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
      dofs_in = 1.0;
      cart3d_mass_operator( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         cart3d_mass_operator( dofs_out, dofs_in );
         cart3d_mass_operator( dofs_in, dofs_out );
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
      cart3d_throughput = num_dofs / time_per_iter;
      std::cout << "Throughput:" << cart3d_throughput << "Dofs/s\n";
   }
#ifdef GENDIL_USE_MFEM
   {
      std::cout << "\n Unstructured 3d mesh mass operator. \n";
      const Integer num_dofs = unstruct_fe_space.GetNumberOfFiniteElementDofs();
      mfem::Vector dofs_in( num_dofs );
      mfem::Vector dofs_out( num_dofs );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = unstruct_fe_space.GetNumberOfFiniteElements();
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
      dofs_in = 1.0;
      unstruct_mass_operator.Mult( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         unstruct_mass_operator.Mult( dofs_out, dofs_in );
         unstruct_mass_operator.Mult( dofs_in, dofs_out );
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
      unstruct_throughput = num_dofs / time_per_iter;
      std::cout << "Throughput:" << unstruct_throughput << "Dofs/s\n";
   }
   {
      std::cout << "\n MFEM mass operator. \n";
      mfem::GridFunction dofs_in( &mfem_fespace );
      mfem::GridFunction dofs_out( &mfem_fespace );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = unstruct_fe_space.GetNumberOfFiniteElements();
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
#endif

   std::cout << "\nSpeedup struct/speed of light: " << struct_throughput / sof_throughput << "\n";

#ifdef GENDIL_USE_MFEM
   std::cout << "\nSpeedup struct/unstruct: " << struct_throughput / unstruct_throughput << "\n";
   std::cout << "\nSpeedup unstruct/mfem: " << unstruct_throughput / mfem_throughput << "\n";
#endif

   // dofs_out.Print( mfem::out, num_elem_dofs );

   return 0;
}
