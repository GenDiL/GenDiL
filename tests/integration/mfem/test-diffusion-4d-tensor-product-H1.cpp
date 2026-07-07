// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>

using namespace gendil;
namespace fs = std::filesystem;

namespace
{

template < typename VectorType >
Real AbsoluteL2Error( const VectorType & a, const VectorType & b )
{
   GENDIL_VERIFY( a.Size() == b.Size(), "Vector sizes do not match." );

   Real err_sq = 0.0;
   for ( Integer i = 0; i < a.Size(); ++i )
   {
      const Real d = a[i] - b[i];
      err_sq += d * d;
   }
   return std::sqrt( err_sq );
}

template < typename VectorType >
Real RelativeL2Error( const VectorType & a, const VectorType & b )
{
   const Real abs_err = AbsoluteL2Error( a, b );

   Real norm_b_sq = 0.0;
   for ( Integer i = 0; i < b.Size(); ++i )
   {
      norm_b_sq += b[i] * b[i];
   }

   const Real norm_b = std::sqrt( norm_b_sq );
   if ( norm_b == 0.0 )
   {
      return abs_err;
   }
   return abs_err / norm_b;
}

template < typename VectorType >
Real L2Norm( const VectorType & x )
{
   Real norm_sq = 0.0;
   for ( Integer i = 0; i < x.Size(); ++i )
   {
      norm_sq += x[i] * x[i];
   }
   return std::sqrt( norm_sq );
}

void FillDeterministicInput( Vector & x )
{
   Real * data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      data[i] =
         0.125 +
         0.03125 * static_cast< Real >( i + 1 ) +
         0.05 * static_cast< Real >( ( 13 * i + 5 ) % 17 );
   }
}

template < Integer order >
int TestTensorProductH1Diffusion4D()
{
   std::cout << "\n=== 4D tensor-product H1 diffusion test, order = "
             << order << " ===\n";

   constexpr Integer factor_dim = 2;
   constexpr Integer dim = 4;

   fs::path base_path = MFEM_DIR;
   fs::path mesh_path = "data/square-disc-nurbs.mesh";
   fs::path file_path = base_path / mesh_path;
   std::cout << "mesh file: " << file_path << "\n";

   const int generate_edges = 0;
   const int refine = 0;
   mfem::Mesh mfem_mesh0( file_path.string().c_str(), generate_edges, refine );

   constexpr Integer unstructured_mesh_order = 2;
   mfem_mesh0.SetCurvature( unstructured_mesh_order );
   QuadMesh< unstructured_mesh_order > mesh0 =
      MakeQuadMesh< unstructured_mesh_order >( mfem_mesh0 );

   const Integer nx1 = 2;
   const Integer ny1 = 2;
   const Real width1 = 1.25;
   const Real height1 = 0.75;
   auto mfem_mesh1 =
      mfem::Mesh::MakeCartesian2D(
         nx1,
         ny1,
         mfem::Element::Type::QUADRILATERAL,
         false,
         width1,
         height1,
         false );

   constexpr Integer cartesian_mesh_order = 1;
   QuadMesh< cartesian_mesh_order > mesh1 =
      MakeQuadMesh< cartesian_mesh_order >( mfem_mesh1 );

   FiniteElementOrders< order, order > factor_orders;
   auto factor_fe = MakeLobattoFiniteElement( factor_orders );

   mfem::H1_FECollection fec0( order, factor_dim );
   mfem::H1_FECollection fec1( order, factor_dim );
   mfem::FiniteElementSpace mfem_fes0( &mfem_mesh0, &fec0 );
   mfem::FiniteElementSpace mfem_fes1( &mfem_mesh1, &fec1 );

   auto factor_restriction0 = GetH1Restriction( mfem_fes0 );
   auto factor_restriction1 = GetH1Restriction( mfem_fes1 );
   auto factor_space0 =
      MakeFiniteElementSpace( mesh0, factor_fe, factor_restriction0 );
   auto factor_space1 =
      MakeFiniteElementSpace( mesh1, factor_fe, factor_restriction1 );
   auto product_mesh = MakeCartesianProductMesh( mesh0, mesh1 );

   FiniteElementOrders< order, order, order, order > product_orders;
   auto product_fe = MakeLobattoFiniteElement( product_orders );
   auto product_restriction =
      MakeTensorProductRestriction( factor_space0, factor_space1 );
   using ProductRestriction =
      std::remove_cvref_t< decltype( product_restriction ) >;
   static_assert( is_tensor_product_restriction_v< ProductRestriction > );
   static_assert(
      restriction_traits< ProductRestriction >::is_direct_index_map );
   static_assert(
      !restriction_traits< ProductRestriction >::is_injective );

   auto fe_space =
      MakeFiniteElementSpace( product_mesh, product_fe, product_restriction );

   const Integer expected_num_dofs =
      mfem_fes0.GetNDofs() * mfem_fes1.GetNDofs();
   const Integer num_dofs = fe_space.GetNumberOfFiniteElementDofs();
   if ( num_dofs != expected_num_dofs )
   {
      std::cerr << "FAILED: tensor-product H1 DoF count mismatch. got "
                << num_dofs << ", expected " << expected_num_dofs << "\n";
      return 1;
   }

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints<
      num_quad_1d,
      num_quad_1d,
      num_quad_1d,
      num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   using KernelPolicy = SerialKernelConfiguration;

   auto diffusivity_fn =
      [] GENDIL_HOST_DEVICE ( const std::array< Real, dim > & X ) -> Real
   {
      return 1.0
         + 0.20 * X[0]
         + 0.13 * X[1] * X[1]
         + 0.11 * X[2]
         + 0.07 * X[3] * X[0]
         + 0.05 * X[1] * X[2];
   };

   Cells< "mesh" > cells;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;

   auto diffusivity =
      MakeCoefficient< "diffusivity", PhysicalCoordinate >( diffusivity_fn );
   auto weak_form =
      integrate( cells, diffusivity * dot( grad( u ), grad( v ) ) );

   auto weak_form_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   auto matrix_free_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         weak_form_context,
         integration_rule );
   auto coo_matrix =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         weak_form_context,
         integration_rule );

   Vector x( num_dofs );
   Vector y_matrix_free( num_dofs );
   Vector y_coo( num_dofs );

   FillDeterministicInput( x );
   y_matrix_free = 0.0;
   y_coo = 0.0;

   matrix_free_operator( x, y_matrix_free );
   coo_matrix( x, y_coo );

   const Real abs_error = AbsoluteL2Error( y_matrix_free, y_coo );
   const Real rel_error = RelativeL2Error( y_matrix_free, y_coo );

   std::cout << "4D H1 DoFs       = " << num_dofs << "\n";
   std::cout << "4D elements      = " << fe_space.GetNumberOfFiniteElements() << "\n";
   std::cout << "||matrix-free||  = " << L2Norm( y_matrix_free ) << "\n";
   std::cout << "||COO||          = " << L2Norm( y_coo ) << "\n";
   std::cout << "abs L2 error     = " << abs_error << "\n";
   std::cout << "rel L2 error     = " << rel_error << "\n";

   constexpr Real tolerance = 1.0e-10;
   if ( rel_error > tolerance )
   {
      std::cerr << "FAILED: 4D tensor-product H1 diffusion action mismatch.\n";
      return 1;
   }

   std::cout << "SUCCESS: 4D tensor-product H1 diffusion order "
             << order << " passed.\n";
   return 0;
}

} // namespace

int main( int, char ** )
{
#if defined(GENDIL_USE_CUDA)
   const char device_config[] = "cuda";
#elif defined(GENDIL_USE_HIP)
   const char device_config[] = "hip";
#else
   const char device_config[] = "cpu";
#endif
   mfem::Device device( device_config );
   device.Print();

   if ( TestTensorProductH1Diffusion4D< 1 >() != 0 ) { return 1; }
   if ( TestTensorProductH1Diffusion4D< 2 >() != 0 ) { return 1; }

   std::cout << "\nAll 4D tensor-product H1 diffusion tests passed.\n";
   return 0;
}
