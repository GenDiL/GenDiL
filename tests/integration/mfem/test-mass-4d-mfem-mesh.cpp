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
   fs::path mesh_path = "data/star-q3.mesh";
   fs::path file_path = base_path / mesh_path;
   std::cout << "mesh file: " << file_path << std::endl;
   const int generate_edges = 0;
   const int refine = 0;
   mfem::Mesh mfem_unstructured_mesh( file_path.string().c_str(), generate_edges, refine );
   constexpr Integer mesh_order = 3;
   mfem_unstructured_mesh.SetCurvature( mesh_order );
   QuadMesh< mesh_order > mesh_1 = MakeQuadMesh< mesh_order >( mfem_unstructured_mesh );

   const Real domain = 1.2345;
   const Integer num_elem_1d = 5;
   const Real h = domain / num_elem_1d;
   const Integer nx = num_elem_1d;
   const Integer ny = num_elem_1d;
   Cartesian2DMesh mesh_2( h, nx, ny );

   auto mesh = MakeCartesianProductMesh( mesh_1, mesh_2 );

   ///////////////////////
   // Finite Element Space

   constexpr Integer order = 1;

   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   constexpr Integer order_W = order;
   FiniteElementOrders< order_X, order_Y, order_Z, order_W > orders;

   // Finite element
   auto finite_element = MakeLobattoFiniteElement( orders );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   ///////////////////
   // Integration Rule

   constexpr Integer num_quad_1d = order+2;

   // Number of quadrature points
   constexpr Integer num_quad_X = num_quad_1d;
   constexpr Integer num_quad_Y = num_quad_1d;
   constexpr Integer num_quad_Z = num_quad_1d;
   constexpr Integer num_quad_W = num_quad_1d;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z, num_quad_W > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      const Real w = X[3];
      return x*y*z*w;
   };

   // Kernel configuration
#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = Dim;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d>;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d,num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   auto mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( fe_space, int_rules, sigma );

   FiniteElementVector dofs_in( fe_space );
   FiniteElementVector dofs_out_mf( fe_space );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   const Integer num_dofs = num_elem * num_elem_dofs;
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   dofs_in.Randomize(1);
   // dofs_in = 0.0;
   // dofs_in[0] = 1.0;
   mass_operator.Mult( dofs_in, dofs_out_mf );

   std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   // std::cout << "in vector: ";
   // dofs_in.Print( std::cout );
   // std::cout << "\nmf vector:\n";
   // dofs_out_mf.Print( std::cout );

   return 0;
}
