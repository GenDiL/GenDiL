// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <chrono>

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

   std::cout << "MFEM_DIR: " << MFEM_DIR << std::endl; 
   fs::path base_path = MFEM_DIR;
   fs::path mesh_path = "data/square-disc-nurbs.mesh";
   fs::path file_path = base_path / mesh_path;
   std::cout << "mesh file: " << file_path << std::endl;
   const int generate_edges = 0;
   const int refine = 0;
   mfem::Mesh mfem_mesh( file_path.string().c_str(), generate_edges, refine );
   constexpr Integer mesh_order = 2;
   const int ref_levels = 0;
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mfem_mesh.UniformRefinement();
   }
   mfem_mesh.SetCurvature( mesh_order );
   QuadMesh< mesh_order > mesh = MakeQuadMesh< mesh_order >( mfem_mesh );
   const int dim = mfem_mesh.Dimension();

   ///////////////////////
   // Finite Element Space

   constexpr Integer order = 2;

   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   FiniteElementOrders< order_X, order_Y > orders;

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
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y > num_quads;

   // High-dimension integration rule
   auto int_rules = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( fe_space );
   auto adv = [=] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X, Real (&v)[Dim] )
   {
      v[0] = X[1];
      v[1] = 1.2345;
   };
   auto zero = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X ){ return 0.0; };

   #if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = Dim;
   // using ThreadLayout = ThreadBlockLayout<num_quad_1d>;
   using ThreadLayout = ThreadBlockLayout<num_quad_1d,num_quad_1d>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
   #else
   using KernelPolicy = SerialKernelConfiguration;
   #endif

   auto advection_operator = MakeAdvectionOperator< KernelPolicy >( fe_space, int_rules, adv, zero );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector dofs_in( num_dofs );
   mfem::Vector dofs_out_mf( num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   dofs_in.Randomize(1);
   // dofs_in = 0.0;
   // dofs_in[7] = 1.0;
   // dofs_in = 1.0;
   advection_operator.Mult( dofs_in, dofs_out_mf );

   /// Using MFEM partial assembly DG
   // auto non_periodic_mesh = mfem::Mesh::MakeCartesian3D( num_elem_1d, num_elem_1d, num_elem_1d, mfem::Element::Type::HEXAHEDRON, 1.0, 1.0, 1.0, false );
   // // Create translation vectors defining the periodicity
   // Vector x_translation({1.0, 0.0, 0.0});
   // Vector y_translation({0.0, 1.0, 0.0});
   // Vector z_translation({0.0, 0.0, 1.0});
   // std::vector<Vector> translations = {x_translation, y_translation, z_translation};
   // // Create the periodic mesh using the vertex mapping defined by the translation vectors
   // Mesh periodic_mesh = Mesh::MakePeriodic(non_periodic_mesh, non_periodic_mesh.CreatePeriodicVertexMapping(translations));
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&mfem_mesh, &fec);
   
   GridFunction dofs_out_pa( &fes );

   auto adv_func = [=](const mfem::Vector& x, mfem::Vector& v)
   {
      v[0] = x[1];
      v[1] = 1.2345;
   };
   VectorFunctionCoefficient velocity(2, adv_func);

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   constexpr double alpha = 1.0; // Should discretize div(adv*psi)
   blf_pa.AddDomainIntegrator(
      new ConservativeConvectionIntegrator(velocity, alpha));
   blf_pa.AddInteriorFaceIntegrator(
      new ConservativeDGTraceIntegrator(velocity, alpha));
   blf_pa.AddBdrFaceIntegrator(
      new ConservativeDGTraceIntegrator(velocity, alpha));
   blf_pa.Assemble();
   blf_pa.Mult(dofs_in, dofs_out_pa);

   std::cout << "dofs_in size: " << dofs_in.Size() << std::endl;
   std::cout << "dofs_out_mf size: " << dofs_out_mf.Size() << std::endl;
   std::cout << "dofs_out_pa size: " << dofs_out_pa.Size() << std::endl;
   std::cout << "Norm of mf: " << dofs_out_mf.Norml2() << std::endl;
   std::cout << "Norm of pa: " << dofs_out_pa.Norml2() << std::endl;

   std::cout << "min(Apsi_mf), max(Apsi_mf): " << dofs_out_mf.Min() << ", " << dofs_out_mf.Max() << endl;
   std::cout << "min(Apsi_pa), max(Apsi_pa): " << dofs_out_pa.Min() << ", " << dofs_out_pa.Max() << endl;
   std::cout << "in vector: ";
   dofs_in.Print( std::cout, num_elem_dofs );
   std::cout << "\nmf vector:\n";
   dofs_out_mf.Print( std::cout, num_elem_dofs );
   std::cout << "\npa vector:\n";
   dofs_out_pa.Print( std::cout, num_elem_dofs );
   // for( size_t i=0; i < dofs_out_mf.Size(); i++)
   // {
   //    const Real diff = dofs_out_mf[i] - dofs_out_pa[i];
   //    const Real distance = diff * diff;
   //    if( distance > 1e-10 )
   //    {
   //       std::cout << i << std::endl;
   //    }
   // }

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
