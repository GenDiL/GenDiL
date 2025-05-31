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
   const Real Lz = 1.0;
   const Integer num_elem_1d = 3;
   const Real h = Lz/num_elem_1d;

   /////////
   // mesh 1
   const Real h_1 = h;
   const Integer n1 = num_elem_1d;
   Cartesian1DMesh mesh_1( h_1, n1 );

   /////////
   // mesh 2
   const Real h_2 = h;
   const Integer n2 = num_elem_1d;
   Cartesian1DMesh mesh_2( h_2, n2 );

   /////////
   // mesh 3
   const Real h_3 = h;
   const Integer n3 = num_elem_1d;
   Cartesian1DMesh mesh_3( h_3, n3 );

   /////////////////////////
   // Cartesian Product Mesh
   auto mesh = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

   ///////////////////////
   // Finite Element Space

   constexpr Integer order = 2;

   // Orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders;

   // Finite element
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto vector_finite_element = MakeVectorFiniteElement( finite_element, finite_element, finite_element );

   // Finite element space
   auto fe_space = MakeFiniteElementSpace( mesh, vector_finite_element );

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

   constexpr Integer Dim = GetDim( fe_space );
   auto sigma = [=] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      return 1.2345;
   };

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
   constexpr size_t NumSharedDimensions = 2;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   const char device_config[] = "cpu";
   using KernelPolicy = SerialKernelConfiguration;
#endif
   mfem::Device device(device_config);
   device.Print();

   auto mass_operator = MakeMassFiniteElementOperator< KernelPolicy >( fe_space, int_rules, sigma );
   // auto mass_operator = MakeSpeedOfLightOperator< KernelPolicy >( fe_space, int_rules );

   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector dofs_in( num_dofs );
   mfem::Vector dofs_out_mf( num_dofs );

   const Integer num_elem_dofs = finite_element.GetNumDofs();
   const Integer num_elem = fe_space.GetNumberOfFiniteElements();
   std::cout << "\n Dofs per element: " << num_elem_dofs << "\n Number of elements: " << num_elem << "\n";
   std::cout << "Total number of dofs: " << num_dofs << "\n";

   dofs_in.Randomize(1);
   mass_operator.Mult( dofs_in, dofs_out_mf );

   // MFEM Legacy operator
   auto mfem_mesh = mfem::Mesh::MakeCartesian3D( n1, n2, n3, mfem::Element::Type::HEXAHEDRON, h*n1, h*n2, h*n3, false );
   int dim = mfem_mesh.Dimension();
   int vdim = dim;
   L2_FECollection fec(order, dim, BasisType::GaussLobatto);
   mfem::FiniteElementSpace fes(&mfem_mesh, &fec, vdim);

   const mfem::IntegrationRule *ir = nullptr;
   
   GridFunction dofs_out_pa( &fes );

   ConstantCoefficient sigma_cf( 1.2345 );

   BilinearForm blf_pa(&fes);
   blf_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf_pa.AddDomainIntegrator(new VectorMassIntegrator(sigma_cf,ir));
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
