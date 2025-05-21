// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <chrono>

using namespace std;
using namespace gendil;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);
   const Real Lz = 1.0;
   const Integer num_elem_1d = 4;
   const Real h = Lz/num_elem_1d;
   const Real h_1 = h;
   const Integer n1 = 8;//num_elem_1d;
   const Real h_2 = h;
   const Integer n2 = 1;//num_elem_1d;
   const Real h_3 = h;
   const Integer n3 = 1;//num_elem_1d;

   Point< 3 > origin{ 0.0, 0.0, 0.0 };
   std::array< int, 3 > periodic_boundarys = { false, false, false };
   int size;
   MPI_Comm_size( MPI_COMM_WORLD, &size );
   std::array< int, 3 > partition = { size, 1, 1 };
   // std::array< Real, 3 > delta_h = { h_1/partition[0], h_2/partition[1], h_3/partition[2] };
   std::array< Real, 3 > serial_h = { h_1, h_2, h_3 };
   std::array< Real, 3 > parallel_h = { h_1, h_2, h_3 };
   std::array< GlobalIndex, 3 > serial_sizes{ n1, n2, n3 };
   std::array< GlobalIndex, 3 > parallel_sizes{ n1/partition[0], n2/partition[1], n3/partition[2] };
   ParallelCartesianMesh< 3 > parallel_mesh( parallel_sizes, parallel_h, origin, periodic_boundarys, partition );
   CartesianMesh< 3 > serial_mesh( serial_sizes, serial_h, origin, periodic_boundarys );

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
   auto parallel_fe_space = MakeFiniteElementSpace( parallel_mesh, finite_element );
   auto serial_fe_space = MakeFiniteElementSpace( serial_mesh, finite_element );

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

   constexpr Integer Dim = GetDim( parallel_fe_space );
   auto adv = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim> & X, Real (&v)[Dim] )
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      v[0] = 1.0;//y;
      v[1] = 1.0;//z;
      v[2] = 1.0;//x;
   };

   constexpr size_t NumSharedDimensions = 2;
#if defined(GENDIL_USE_DEVICE)
   #if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
   #elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
   #endif
   // using ThreadLayout = ThreadBlockLayout<>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
#else
   const char device_config[] = "cpu";
   using ThreadLayout = ThreadBlockLayout<>;
#endif
   mfem::Device device(device_config);
   device.Print();
   using ThreadingPolicy = ThreadFirstKernelConfiguration<ThreadLayout, NumSharedDimensions >;

   auto parallel_adv_operator = MakeAdvectionOperator< ThreadingPolicy >( parallel_fe_space, int_rules, adv );
   auto serial_adv_operator = MakeAdvectionOperator< ThreadingPolicy >( serial_fe_space, int_rules, adv );

   const Integer parallel_num_dofs = parallel_fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector parallel_dofs_in( parallel_num_dofs );
   mfem::Vector parallel_dofs_out_mf( parallel_num_dofs );
   const Integer serial_num_dofs = serial_fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector serial_dofs_in( serial_num_dofs );
   mfem::Vector serial_dofs_out_mf( serial_num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer parallel_num_elem = parallel_fe_space.GetNumberOfFiniteElements();
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Parallel number of elements: " << parallel_num_elem << "\n";
   std::cout << "Parallel total number of dofs: " << parallel_num_dofs << "\n";
   const Integer serial_num_elem = serial_fe_space.GetNumberOfFiniteElements();
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Serial number of elements: " << serial_num_elem << "\n";
   std::cout << "Serial total number of dofs: " << serial_num_dofs << "\n";

   // dofs_in.Randomize(1);
   // dofs_in.HostReadWrite();
   int my_rank;
   MPI_Comm_rank( parallel_mesh.halo.communicator, &my_rank );
   serial_dofs_in = 1.0;
   // serial_dofs_in[0] = 1.0;
   serial_adv_operator.Mult( serial_dofs_in, serial_dofs_out_mf );
   if( my_rank == 0 )
   {
      std::cout << "\nserial vector in:\n";
      serial_dofs_in.Print( std::cout );
      std::cout << "\nserial vector:\n";
      serial_dofs_out_mf.Print( std::cout );
   }
   parallel_dofs_in = 1.0;
   // if( my_rank == 0 ) parallel_dofs_in[0] = 1.0;
   std::cout << "Starting parallel computations\n";
   std::cout << "\nparallel vector "<< my_rank << " input:\n";
   parallel_dofs_in.Print( std::cout );
   parallel_adv_operator.Mult( parallel_dofs_in, parallel_dofs_out_mf );
   std::cout << "\nparallel vector "<< my_rank << ":\n";
   parallel_dofs_out_mf.Print( std::cout );

   // bool correct = true;
   // for (size_t i = 0; i < parallel_num_dofs; i++)
   // {
   //    const Real par_val = parallel_dofs_out_mf[i];
   //    const Real ser_val = serial_dofs_out_mf[i + my_rank * parallel_num_dofs ];
   //    if ( par_val != ser_val )
   //    {
   //       std::cout << "p_" << my_rank << "[" << i << "] != s[" << i << "] => " << par_val << " vs " << ser_val << std::endl;
   //       correct = false;
   //    }
   //    else
   //    {
   //       std::cout << "p_" << my_rank << "[" << i << "] == s[" << i << "]" << std::endl;
   //    }      
   // }
   

   // MFEM Legacy operator
   // auto mfem_parallel_mesh = mfem::Mesh::MakeCartesian3D( n1, n2, n3, mfem::Element::Type::HEXAHEDRON, h*n1, h*n2, h*n3, false );
   // int dim = mfem_parallel_mesh.Dimension();
   // mfem::L2_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
   // mfem::FiniteElementSpace fes(&mfem_parallel_mesh, &fec);

   // const mfem::IntegrationRule *ir = nullptr;
   
   // mfem::GridFunction dofs_out_pa( &fes );

   // auto adv_func = [=](const mfem::Vector& x, mfem::Vector& v)
   // {
   //    const Real X = x(0);
   //    const Real Y = x(1);
   //    const Real Z = x(2);
   //    v[0] = Y;
   //    v[1] = Z;
   //    v[2] = X;
   // };
   // mfem::VectorFunctionCoefficient velocity(3, adv_func);

   // mfem::BilinearForm blf_pa(&fes);
   // blf_pa.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   // constexpr double alpha = 1.0; // Should discretize div(adv*psi)
   // blf_pa.AddDomainIntegrator(
   //    new mfem::ConservativeConvectionIntegrator(velocity, alpha));
   // blf_pa.AddInteriorFaceIntegrator(
   //    new mfem::ConservativeDGTraceIntegrator(velocity, alpha));
   // blf_pa.AddBdrFaceIntegrator(
   //    new mfem::ConservativeDGTraceIntegrator(velocity, alpha));
   // blf_pa.Assemble();
   // blf_pa.Mult(dofs_in, dofs_out_pa);

   // std::cout << "dofs_out_mf dot: "<< mfem::InnerProduct( parallel_mesh.communicator , dofs_out_mf, dofs_out_mf ) << std::endl;
   // std::cout << "dofs_out_pa dot: "<< mfem::InnerProduct( dofs_out_pa, dofs_out_pa ) << std::endl;

   // std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   // std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   // std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   // std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   // std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;
   // std::cout << "in vector: ";
   // dofs_in.Print( std::cout );
   // std::cout << "\nmf vector:\n";
   // dofs_out_mf.Print( std::cout );
   // std::cout << "\npa vector:\n";
   // dofs_out_pa.Print( std::cout );

   // dofs_out_mf -= dofs_out_pa;

   // const Real absolute_error = dofs_out_mf.Norml2();
   // const Real relative_error = dofs_out_mf.Norml2()/dofs_out_pa.Norml2();
   // std::cout << "Absolute norm of the diff: " << absolute_error << std::endl;
   // std::cout << "Relative norm of the diff: " << relative_error << std::endl;
   
#ifdef GENDIL_USE_MPI
   MPI_Finalize();
#endif

   // const Real tolerance = 1e-6;
   // if( relative_error > tolerance )
   // {
   //    return 1;
   // }

   return 0;
}
