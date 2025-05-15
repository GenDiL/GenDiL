// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <filesystem>
namespace fs = std::filesystem;

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

   ///////
   // Mesh
   std::cout << "MFEM_DIR: " << MFEM_DIR << std::endl; 
   fs::path base_path = MFEM_DIR;
   // fs::path mesh_path = "data/fichera-q3.mesh";
   // fs::path mesh_path = "data/ball-nurbs.mesh";
   fs::path mesh_path = "data/pipe-nurbs.mesh";
   fs::path file_path = base_path / mesh_path;
   std::cout << "mesh file: " << file_path << std::endl;
   const int generate_edges = 0;
   const int refine = 0;
   mfem::Mesh mfem_unstructured_mesh( file_path.string().c_str(), generate_edges, refine );
   constexpr Integer mesh_order = 3; // !FIXME This is a problem currently on GPU -> Optimize mesh interpolations
   mfem_unstructured_mesh.SetCurvature( mesh_order );
   HexMesh< mesh_order > mesh = MakeHexMesh< mesh_order >( mfem_unstructured_mesh );

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

   constexpr Integer num_quad_1d = order+5;

   // Number of quadrature points
   constexpr Integer num_quad_X = num_quad_1d;
   constexpr Integer num_quad_Y = num_quad_1d;
   constexpr Integer num_quad_Z = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return x*y*z;
   };

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 2;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( fe_space, int_rules, sigma );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector dofs_in( num_dofs );
   mfem::Vector dofs_out_mf( num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   dofs_in.Randomize(1);
   // dofs_in = 0.0;
   // dofs_in[0] = 1.0;
   mass_operator.Mult( dofs_in, dofs_out_mf );

   // MFEM Legacy operator
   int dim = mfem_unstructured_mesh.Dimension();
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&mfem_unstructured_mesh, &fec);

   const mfem::IntegrationRule *ir = nullptr;
   
   GridFunction dofs_out_pa( &fes );

   auto sigma_func = [=](const mfem::Vector& X)
   {
      const Real x = X(0);
      const Real y = X(1);
      const Real z = X(2);
      return x*y*z;
   };
   FunctionCoefficient sigma_cf(sigma_func);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new MassIntegrator(sigma_cf,ir));
   blf_pa.Assemble();
   blf_pa.Mult(dofs_in, dofs_out_pa);

   std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;
   // std::cout << "in vector: ";
   // dofs_in.Print( std::cout );
   // std::cout << "\nmf vector:\n";
   // dofs_out_mf.Print( std::cout );
   // std::cout << "\npa vector:\n";
   // dofs_out_pa.Print( std::cout );

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
