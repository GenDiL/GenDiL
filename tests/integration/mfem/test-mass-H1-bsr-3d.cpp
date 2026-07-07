// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <mfem.hpp>

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

using namespace gendil;

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
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] =
         0.25 +
         0.125 * static_cast< Real >( i + 1 ) +
         static_cast< Real >( ( 7 * i + 3 ) % 11 );
   }
}

template < Integer order >
int TestH1MassSGBSR()
{
   std::cout << "\n=== Scalar H1 cell mass SGBSR test, order = "
             << order << " ===\n";

   constexpr Integer dim = 3;
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );

   Cartesian1DMesh mesh_1( h, n );
   Cartesian1DMesh mesh_2( h, n );
   Cartesian1DMesh mesh_3( h, n );
   auto mesh = MakeCartesianProductMesh( mesh_1, mesh_2, mesh_3 );

   FiniteElementOrders< order, order, order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );

   auto mfem_mesh =
      mfem::Mesh::MakeCartesian3D(
         n,
         n,
         n,
         mfem::Element::Type::HEXAHEDRON,
         1.0,
         1.0,
         1.0,
         false );
   mfem::H1_FECollection fec( order, dim );
   mfem::FiniteElementSpace mfem_fes( &mfem_mesh, &fec );

   auto restriction = GetH1Restriction( mfem_fes );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element, restriction );

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints< num_quad_1d, num_quad_1d, num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   using KernelPolicy = SerialKernelConfiguration;

   auto density_fn =
      [] GENDIL_HOST_DEVICE ( const std::array< Real, dim > & X ) -> Real
   {
      const Real x = X[0];
      const Real y = X[1];
      const Real z = X[2];
      return 1.0 + 0.25 * x + 0.5 * y * y + 0.75 * z;
   };

   Cells< "mesh1" > domain;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;

   auto density = MakeCoefficient< "density", PhysicalCoordinate >( density_fn );
   auto weak_form = integrate( domain, density * u * v );

   auto weak_form_context = MakeWeakFormContext(
      MakeTrialField< "u" >( fe_space ),
      MakeIntegrationDomain< "mesh1" >( fe_space ) );

   auto matrix_free_operator =
      MakeMassFiniteElementOperator< KernelPolicy >(
         fe_space,
         integration_rule,
         density_fn );

   auto sgbsr_matrix =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         weak_form_context,
         integration_rule );

   const Integer ndofs = fe_space.GetNumberOfFiniteElementDofs();
   Vector x( ndofs );
   Vector y_matrix_free( ndofs );
   Vector y_sgbsr( ndofs );

   FillDeterministicInput( x );
   y_matrix_free = 0.0;
   y_sgbsr = 17.0;

   matrix_free_operator( x, y_matrix_free );
   sgbsr_matrix( x, y_sgbsr );

   const Real abs_error = AbsoluteL2Error( y_sgbsr, y_matrix_free );
   const Real rel_error = RelativeL2Error( y_sgbsr, y_matrix_free );

   std::cout << "H1 DoFs         = " << ndofs << "\n";
   std::cout << "BSR trial size  = " << sgbsr_matrix.TrialBsrSize() << "\n";
   std::cout << "BSR test size   = " << sgbsr_matrix.TestBsrSize() << "\n";
   std::cout << "||matrix-free|| = " << L2Norm( y_matrix_free ) << "\n";
   std::cout << "||SGBSR||       = " << L2Norm( y_sgbsr ) << "\n";
   std::cout << "abs L2 error   = " << abs_error << "\n";
   std::cout << "rel L2 error   = " << rel_error << "\n";

   const Real tolerance = 1.0e-12;
   if ( rel_error > tolerance )
   {
      std::cerr
         << "FAILED: scalar H1 cell mass SGBSR action does not match "
         << "matrix-free action.\n";
      return 1;
   }

   std::cout << "SUCCESS: order " << order << " passed.\n";
   return 0;
}

} // namespace

int main()
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

   if ( TestH1MassSGBSR< 1 >() != 0 )
   {
      return 1;
   }

   if ( TestH1MassSGBSR< 2 >() != 0 )
   {
      return 1;
   }

   std::cout << "\nScalar H1 cell mass SGBSR test passed.\n";
   return 0;
}
