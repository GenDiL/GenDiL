// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

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

template <
   typename FiniteElementSpace,
   typename VectorType,
   size_t Component >
void FillComponentInput(
   const FiniteElementSpace & fe_space,
   VectorType & x,
   std::integral_constant< size_t, Component > component )
{
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;
   using ComponentDofShape =
      component_dof_shape_t< ShapeFunctions, Component >;

   Real * x_data = x.WriteHostData();
   const GlobalIndex num_elements = fe_space.GetNumberOfFiniteElements();

   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< ComponentDofShape >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex fe_index =
            GlobalDofIndex( fe_space, component, element, indices );
         const GlobalIndex local_index =
            FlattenLocalDof( fe_space, component, indices );

         x_data[fe_index] =
            1000.0 * static_cast< Real >( Component + 1 ) +
            100.0 * static_cast< Real >( element + 1 ) +
            0.5 * static_cast< Real >( local_index + 1 ) +
            static_cast< Real >( fe_index % 5 );
      });
   }
}

template < typename FiniteElementSpace, typename VectorType >
void FillDeterministicVectorInput(
   const FiniteElementSpace & fe_space,
   VectorType & x )
{
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;

   x = -1.0;

   constexpr Integer vdim = ShapeFunctions::vector_dim;
   ConstexprLoop< vdim >( [&] ( auto component )
   {
      FillComponentInput( fe_space, x, component );
   });
}

template < Integer order >
int TestVectorBoundaryMassBsrAction()
{
   std::cout << "\n=== Vector boundary mass BSR action test, order = "
             << order << " ===\n";

   constexpr Integer Dim = 2;
   const Integer n = 3;
   const Real h = 1.0 / n;
   CartesianMesh< Dim > mesh( { n, n }, { h, h }, { 0.0, 0.0 } );

   FiniteElementOrders< order, order > orders;
   auto scalar_finite_element = MakeLobattoFiniteElement( orders );
   auto vector_finite_element = MakeVectorFiniteElement(
      scalar_finite_element,
      scalar_finite_element );
   auto vector_fe_space = MakeFiniteElementSpace( mesh, vector_finite_element );

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints< num_quad_1d, num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = 1;
   using ThreadLayout = ThreadBlockLayout< num_quad_1d >;
   using KernelPolicy =
      ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   BoundaryFacets< "mesh1" > boundary_facets;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;

   auto weak_form = integrate( boundary_facets, dot( u, v ) );

   auto weak_form_context = MakeWeakFormContext(
      MakeTrialField< "u" >( vector_fe_space ),
      MakeIntegrationDomain< "mesh1" >( vector_fe_space ) );

   auto matrix_free_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         weak_form_context,
         integration_rule );

   auto sgbsr_matrix =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         weak_form_context,
         integration_rule );
   const Integer ndofs = vector_fe_space.GetNumberOfFiniteElementDofs();
   Vector x( ndofs );
   Vector y_matrix_free( ndofs );
   Vector y_bsr( ndofs );

   FillDeterministicVectorInput( vector_fe_space, x );
   y_matrix_free = 0.0;
   y_bsr = 0.0;

   matrix_free_operator( x, y_matrix_free );
   sgbsr_matrix( x, y_bsr );

   const Real abs_error = AbsoluteL2Error( y_bsr, y_matrix_free );
   const Real rel_error = RelativeL2Error( y_bsr, y_matrix_free );

   std::cout << "Vector DoFs     = " << ndofs << "\n";
   std::cout << "BSR trial size  = " << sgbsr_matrix.TrialBsrSize() << "\n";
   std::cout << "BSR test size   = " << sgbsr_matrix.TestBsrSize() << "\n";
   std::cout << "||matrix-free|| = " << L2Norm( y_matrix_free ) << "\n";
   std::cout << "||SGBSR||       = " << L2Norm( y_bsr ) << "\n";
   std::cout << "abs L2 error   = " << abs_error << "\n";
   std::cout << "rel L2 error   = " << rel_error << "\n";

   const Real tolerance = 1.0e-12;
   if ( rel_error > tolerance )
   {
      std::cerr
         << "FAILED: vector boundary mass SGBSR action does not match "
         << "matrix-free action.\n";
      return 1;
   }

   std::cout << "SUCCESS: order " << order << " passed.\n";
   return 0;
}

} // namespace

int main()
{
   if ( TestVectorBoundaryMassBsrAction< 2 >() != 0 )
   {
      return 1;
   }

   std::cout << "\nVector boundary mass BSR action test passed.\n";
   return 0;
}
