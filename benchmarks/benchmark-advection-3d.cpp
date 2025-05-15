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
   const auto gendil_start = std::chrono::steady_clock::now();
   const Integer num_elem_1d = 50;
   /////////
   // mesh 1
   const Real h_1 = 1.0;
   const Integer n1 = num_elem_1d;
   Cartesian1DMesh mesh_1( h_1, n1 );

   /////////
   // mesh 2
   const Real h_2 = 2.0;
   const Integer n2 = num_elem_1d;
   Cartesian1DMesh mesh_2( h_2, n2 );

   /////////
   // mesh 3
   const Real h_3 = 1.0;
   const Integer n3 = num_elem_1d;
   Cartesian1DMesh mesh_3( h_3, n3 );

   Cartesian3DMesh cart3d_mesh( h_1, n1, n2, n3 );

   //////////////////////
   // Tensor product Mesh
   auto mesh = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

   ///////////////////////
   // Finite Element Space
   constexpr Integer order = 5;

   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders;

   // Finite element
   auto finite_element = MakeLobattoFiniteElement( orders );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   auto cart3d_fe_space = MakeFiniteElementSpace( cart3d_mesh, finite_element );

   constexpr Integer Dim = GetDim( fe_space );

   ///////////////////
   // Integration Rule
   constexpr Integer num_quad = order + 2;

   // Number of quadrature points
   constexpr Integer num_quad_X = num_quad;
   constexpr Integer num_quad_Y = num_quad;
   constexpr Integer num_quad_Z = num_quad;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   auto advection_field = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X, Real (&v)[Dim] ) -> void
   {
      v[0] = 1.0;
      v[1] = 1.0;
      v[2] = 1.0;
   };

#if defined(GENDIL_USE_DEVICE)
#ifdef GENDIL_USE_MFEM
   #if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
   #elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
   #endif
#endif
   constexpr Integer NumSharedDimensions = 3;
   // using ThreadLayout = ThreadBlockLayout<num_quad,num_quad>;
   using ThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad>;
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

   auto sol_operator = MakeFaceSpeedOfLightOperator< KernelPolicy >( cart3d_fe_space, int_rules );
   auto adv_operator = MakeAdvectionOperator< KernelPolicy >( fe_space, int_rules, advection_field );
   auto cart3d_adv_operator = MakeAdvectionOperator< KernelPolicy >( cart3d_fe_space, int_rules, advection_field );

   const Integer num_iter = 5;

   std::cout << "\nOrder:" << order << "\n";
   std::cout << "Num quad:" << num_quad << "\n";
   {
      std::cout << "\n Speed-of-light face operator. \n";
      const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = fe_space.GetNumberOfFiniteElements();
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
   }

   {
      std::cout << "\n Cartesian 3d mesh advection operator. \n";
      const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = cart3d_fe_space.GetNumberOfFiniteElements();
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";

      dofs_in = 1.0;
      cart3d_adv_operator( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         cart3d_adv_operator( dofs_out, dofs_in );
         cart3d_adv_operator( dofs_in, dofs_out );
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
   }

   {
      std::cout << "\n Tensor mesh advection operator. \n";
      const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
      Vector dofs_in( num_dofs );
      Vector dofs_out( num_dofs );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = fe_space.GetNumberOfFiniteElements();
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";

      dofs_in = 1.0;
      adv_operator( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         adv_operator( dofs_out, dofs_in );
         adv_operator( dofs_in, dofs_out );
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
   }
   const auto gendil_end = std::chrono::steady_clock::now();
   const std::chrono::duration<double> gendil_elapsed_seconds = gendil_end - gendil_start;
   std::cout << "gendil Setup + Run time:" << gendil_elapsed_seconds.count() << "s\n";

#ifdef GENDIL_USE_MFEM
   // MFEM operator
   const auto mfem_start = std::chrono::steady_clock::now();
   const int dim = 3;
   auto non_periodic_mesh = mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON, 1.0, 1.0, 1.0, false );
   mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
   mfem::FiniteElementSpace mfem_fespace(&non_periodic_mesh, &fec);
   
   auto adv_func = [=](const mfem::Vector& x, mfem::Vector& v)
   {
      v[0] = 1.0;
      v[1] = 1.0;
      v[2] = 1.0;
   };
   mfem::VectorFunctionCoefficient velocity(3, adv_func);

   mfem::BilinearForm mfem_operator(&mfem_fespace);
   mfem_operator.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   constexpr double alpha = 1.0; // Should discretize div(adv*psi)
   mfem_operator.AddDomainIntegrator(
      new mfem::ConservativeConvectionIntegrator(velocity, alpha));
   mfem_operator.AddInteriorFaceIntegrator(
      new mfem::ConservativeDGTraceIntegrator(velocity, alpha));
   mfem_operator.AddBdrFaceIntegrator(
      new mfem::ConservativeDGTraceIntegrator(velocity, alpha));
   mfem_operator.Assemble();

   {
      std::cout << "\n MFEM advection operator. \n";
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
      std::cout << "Throughput:" << num_dofs / time_per_iter << "Dofs/s\n";
   }
   const auto mfem_end = std::chrono::steady_clock::now();
   const std::chrono::duration<double> mfem_elapsed_seconds = mfem_end - mfem_start;
   std::cout << "MFEM Setup + Run time:" << mfem_elapsed_seconds.count() << "s\n";

   const auto unstruct_start = std::chrono::steady_clock::now();
   {
      std::cout << "\n Unstructured 3d mesh advection operator. \n";
      non_periodic_mesh.EnsureNodes();
      HexMesh<1> unstruct_mesh = MakeHexMesh< 1 >( non_periodic_mesh );
      auto fe_space = MakeFiniteElementSpace( unstruct_mesh, finite_element );
      auto adv_operator = MakeAdvectionOperator< KernelPolicy >( fe_space, int_rules, advection_field );
      const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
      mfem::Vector dofs_in( num_dofs );
      mfem::Vector dofs_out( num_dofs );

      const Integer num_elem_dofs = finite_element.GetNumDofs();
      const Integer num_elem = fe_space.GetNumberOfFiniteElements();
      std::cout << "Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";

      dofs_in = 1.0;
      adv_operator.Mult( dofs_in, dofs_out );

      GENDIL_DEVICE_SYNC;
      const auto start = std::chrono::steady_clock::now();
      for ( Integer iter = 0; iter < num_iter; iter++ )
      {
         adv_operator.Mult( dofs_out, dofs_in );
         adv_operator.Mult( dofs_in, dofs_out );
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
   }
   const auto unstruct_end = std::chrono::steady_clock::now();
   const std::chrono::duration<double> unstruct_elapsed_seconds = unstruct_end - unstruct_start;
   std::cout << "Unstruct gendil Setup + Run time:" << unstruct_elapsed_seconds.count() << "s\n";
#endif

   return 0;
}
