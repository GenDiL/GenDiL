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
   const Integer num_elem_1d = 2;
   const Real h = Lz/num_elem_1d;
   const Real h_1 = h;
   const Integer n1 = num_elem_1d;
   const Real h_2 = h;
   const Integer n2 = num_elem_1d;
   const Real h_3 = h;
   const Integer n3 = num_elem_1d;

   Point< 3 > origin{ 0.0, 0.0, 0.0 };
   std::array< int, 3 > periodic_boundarys = { false, false, false };
   int size;
   MPI_Comm_size( MPI_COMM_WORLD, &size );
   std::array< int, 3 > partition = { size, 1, 1 };
   // std::array< Real, 3 > delta_h = { h_1/partition[0], h_2/partition[1], h_3/partition[2] };
   // std::array< GlobalIndex, 3 > sizes{ n1, n2, n3 };
   std::array< Real, 3 > serial_h = { h_1, h_2, h_3 };
   std::array< Real, 3 > parallel_h = { h_1, h_2, h_3 };
   std::array< GlobalIndex, 3 > serial_sizes{ n1, n2, n3 };
   std::array< GlobalIndex, 3 > parallel_sizes{ n1/partition[0], n2/partition[1], n3/partition[2] };
   ParallelCartesianMesh< 3 > parallel_mesh( parallel_sizes, parallel_h, origin, periodic_boundarys, partition );
   CartesianMesh< 3 > serial_mesh( serial_sizes, serial_h, origin, periodic_boundarys );

   ///////////////////////
   // Finite Element Space

   constexpr Integer order = 0;

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
   auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return x*y*z;
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
   using ThreadingPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;

   auto parallel_mass_operator = MakeMassFiniteElementOperator< ThreadingPolicy >( parallel_fe_space, int_rules, sigma );
   auto serial_mass_operator = MakeMassFiniteElementOperator< ThreadingPolicy >( serial_fe_space, int_rules, sigma );

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

   // parallel_dofs_in.Randomize(1);
   parallel_dofs_in = 1.0;
   parallel_mass_operator.Mult( parallel_dofs_in, parallel_dofs_out_mf );
   serial_dofs_in = 1.0;
   serial_mass_operator.Mult( serial_dofs_in, serial_dofs_out_mf );

   // MFEM Legacy operator
   // auto mfem_mesh = mfem::Mesh::MakeCartesian3D( n1, n2, n3, mfem::Element::Type::HEXAHEDRON, h*n1, h*n2, h*n3, false );
   // int dim = mfem_mesh.Dimension();
   // mfem::L2_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
   // mfem::FiniteElementSpace fes(&mfem_mesh, &fec);

   // const mfem::IntegrationRule *ir = nullptr;
   
   // mfem::GridFunction dofs_out_pa( &fes );

   // auto sigma_func = [=](const mfem::Vector& X)
   // {
   //    const Real x = X(0);
   //    const Real y = X(1);
   //    const Real z = X(2);
   //    return x*y*z;
   // };
   // mfem::FunctionCoefficient sigma_cf(sigma_func);

   // mfem::BilinearForm blf_pa(&fes);
   // blf_pa.SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
   // blf_pa.AddDomainIntegrator(new mfem::MassIntegrator(sigma_cf,ir));
   // blf_pa.Assemble();
   // blf_pa.Mult(parallel_dofs_in, dofs_out_pa);

   std::cout << "parallel_dofs_in size: " << parallel_dofs_in.Size() << std::endl;
   std::cout << "parallel_dofs_out_mf size: " << parallel_dofs_out_mf.Size() << std::endl;
   // std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   std::cout << "Parallel Norm of mf: " << parallel_dofs_out_mf.Norml2() << std::endl;
   std::cout << "Serial Norm of mf: " << serial_dofs_out_mf.Norml2() << std::endl;
   std::cout << "parrallel_dofs_out_mf dot: "<< mfem::InnerProduct( parallel_mesh.halo.communicator , parallel_dofs_out_mf, parallel_dofs_out_mf ) << std::endl;
   std::cout << "serial_dofs_out_mf dot: "<< mfem::InnerProduct( serial_dofs_out_mf, serial_dofs_out_mf ) << std::endl;
   // std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;
   // std::cout << "in vector: ";
   // parallel_dofs_in.Print( std::cout );
   std::cout << "\nparallel vector:\n";
   parallel_dofs_out_mf.Print( std::cout );
   std::cout << "\nserial vector:\n";
   serial_dofs_out_mf.Print( std::cout );

   // parallel_dofs_out_mf -= dofs_out_pa;

   // const Real absolute_error = parallel_dofs_out_mf.Norml2();
   // const Real relative_error = parallel_dofs_out_mf.Norml2()/dofs_out_pa.Norml2();
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
