// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <mfem.hpp>
#include <gendil/gendil.hpp>

using namespace std;
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

   const Integer num_elem_1d = 4;

   /////////////////////////////////
   // Gendil periodic Cartesian mesh
   const Integer n_x = num_elem_1d;
   const Real h_x = 1.0/n_x;
   const Integer n_y = 2*num_elem_1d;
   const Real h_y = 1.0/n_y;
   const Integer n_z = 3*num_elem_1d;
   const Real h_z = 1.0/n_z;
   Point< 3 > origin { 0.0, 0.0, 0.0 };
   PeriodicCartesian3DMesh struct_mesh( h_x, h_y, h_z, n_x, n_y, n_y, origin );

   /////////////////////////////////
   // MFEM periodic Cartesian mesh
   auto non_periodic_mesh = mfem::Mesh::MakeCartesian3D( n_x, n_y, n_z, mfem::Element::Type::HEXAHEDRON, 1.0, 1.0, 1.0, false );
   // Create translation vectors defining the periodicity
   mfem::Vector x_translation({1.0, 0.0, 0.0});
   mfem::Vector y_translation({0.0, 1.0, 0.0});
   mfem::Vector z_translation({0.0, 0.0, 1.0});
   std::vector<mfem::Vector> translations = {x_translation, y_translation, z_translation};
   // Create the periodic mesh using the vertex mapping defined by the translation vectors
   mfem::Mesh periodic_mesh = mfem::Mesh::MakePeriodic(non_periodic_mesh, non_periodic_mesh.CreatePeriodicVertexMapping(translations));
   constexpr Integer mesh_order = 1;
   HexMesh< mesh_order > unstruct_mesh = MakeHexMesh< mesh_order >( periodic_mesh );

   ///////////////////////
   // Finite Element Space
   constexpr Integer order = 0;

   // Space orders
   constexpr Integer order_X = order;
   constexpr Integer order_Y = order;
   constexpr Integer order_Z = order;
   FiniteElementOrders< order_X, order_Y, order_Z > orders;

   // Finite elements
   auto finite_element = MakeLegendreFiniteElement( orders );

   // Finite element spaces
   auto struct_fe_space = MakeFiniteElementSpace( struct_mesh, finite_element );
   auto unstruct_fe_space = MakeFiniteElementSpace( unstruct_mesh, finite_element );

   ///////////////////
   // Integration Rule
   constexpr Integer num_quad = 2;

   // Space number of quadrature points
   constexpr Integer num_quad_X = order_X + num_quad;
   constexpr Integer num_quad_Y = order_Y + num_quad;
   constexpr Integer num_quad_Z = order_Z + num_quad;
   IntegrationRuleNumPoints< num_quad_X, num_quad_Y, num_quad_Z > num_quads;

   // High-dimension integration rule
   auto int_rule = MakeIntegrationRule( num_quads );

   constexpr Integer Dim = GetDim( struct_fe_space );
   auto f = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      return 1.0;
   };

#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 3;
   // using ThreadLayout = ThreadBlockLayout<num_quad,num_quad>;
   using ThreadLayout = ThreadBlockLayout<num_quad,num_quad,num_quad>;
   using KernelPolicy = ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   std::cout << "Computing struct linear form\n";
   auto struct_rhs_g = MakeLinearForm< KernelPolicy >( struct_fe_space, int_rule, f);
   std::cout << "Computing unstruct linear form\n";
   auto unstruct_rhs_g = MakeLinearForm< KernelPolicy >( unstruct_fe_space, int_rule, f);

   mfem::Vector struct_rhs = struct_rhs_g.ToMFEMVector();
   mfem::Vector unstruct_rhs = unstruct_rhs_g.ToMFEMVector();
   struct_rhs -= unstruct_rhs;

   const double tol = 1e-10;
   const double norm = struct_rhs.Norml2();
   bool success = false;
   if (norm < tol)
   {
      std::cout << "SUCCESS: " << norm << std::endl;
      success = true;
   }
   else
   {
      std::cout << "FAILED: " << norm << std::endl;
   }

   auto advection_field = [] GENDIL_HOST_DEVICE ( const std::array< Real, Dim> & X, Real (&v)[Dim] )
   {
      v[0] = 1.0;
      v[1] = 2.0;
      v[2] = 3.0;
   };

   auto boundary_field = [] GENDIL_HOST_DEVICE ( std::array< Real, Dim> const & X ) -> Real
   {
      return 1.0;
   };

   auto struct_adv_operator_with_bc = MakeAdvectionOperator< KernelPolicy >( struct_fe_space, int_rule, advection_field, boundary_field );
   auto unstruct_adv_operator_with_bc = MakeAdvectionOperator< KernelPolicy >( unstruct_fe_space, int_rule, advection_field, boundary_field );

   const Integer num_dofs_struct = struct_fe_space.GetNumberOfFiniteElementDofs();
   const Integer num_dofs_unstruct = unstruct_fe_space.GetNumberOfFiniteElementDofs();
   mfem::Vector struct_v( num_dofs_struct );
   mfem::Vector unstruct_v( num_dofs_unstruct );

   std::cout << "Computing struct advection\n";
   struct_adv_operator_with_bc.Mult( unstruct_rhs, struct_v );
   std::cout << "Computing unstruct advection\n";
   unstruct_adv_operator_with_bc.Mult( unstruct_rhs, unstruct_v );

   struct_v -= unstruct_v;

   const double norm_v = struct_v.Norml2();
   bool success_v = false;
   if (norm_v < tol)
   {
      std::cout << "SUCCESS: " << norm_v << std::endl;
      success_v = true;
   }
   else
   {
      std::cout << "FAILED: " << norm_v << std::endl;
   }

   return (success && success_v) ? 0 : 1;
}
