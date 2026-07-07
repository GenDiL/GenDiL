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
            0.125 * static_cast< Real >( local_index + 1 ) +
            static_cast< Real >( ( 3 * fe_index + 1 ) % 11 );
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

template < Integer order, bool UseAnisotropicComponents >
int TestVectorInteriorJumpBsrAction()
{
   std::cout << "\n=== Vector interior jump BSR action test, order = "
             << order
             << ", "
             << ( UseAnisotropicComponents ? "anisotropic" : "isotropic" )
             << " components ===\n";

   constexpr Integer Dim = 2;
   const Integer n = 3;
   const Real h = 1.0 / n;
   CartesianMesh< Dim > mesh( { n, n }, { h, h }, { 0.0, 0.0 }, true );

   auto vector_finite_element = []()
   {
      if constexpr ( UseAnisotropicComponents )
      {
         FiniteElementOrders< order, order > component0_orders;
         FiniteElementOrders< order + 1, order > component1_orders;
         auto component0_finite_element =
            MakeLobattoFiniteElement( component0_orders );
         auto component1_finite_element =
            MakeLobattoFiniteElement( component1_orders );
         return MakeVectorFiniteElement(
            component0_finite_element,
            component1_finite_element );
      }
      else
      {
         FiniteElementOrders< order, order > orders;
         auto scalar_finite_element = MakeLobattoFiniteElement( orders );
         return MakeVectorFiniteElement(
            scalar_finite_element,
            scalar_finite_element );
      }
   }();
   auto vector_fe_space = MakeFiniteElementSpace( mesh, vector_finite_element );

   constexpr Integer num_quad_1d = order + 3;
   IntegrationRuleNumPoints< num_quad_1d, num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

#if defined(GENDIL_USE_DEVICE)
   constexpr Integer NumSharedDimensions = Dim;
   using ThreadLayout = ThreadBlockLayout< num_quad_1d, num_quad_1d >;
   using KernelPolicy =
      ThreadFirstKernelConfiguration< ThreadLayout, NumSharedDimensions >;
#else
   using KernelPolicy = SerialKernelConfiguration;
#endif

   InteriorFacets< "mesh1" > interior_facets;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;

   auto tau = MakeCoefficient< "penalty" >(
      [] GENDIL_HOST_DEVICE () -> Real
      {
         return 2.75;
      });

   auto weak_form =
      integrate(
         interior_facets,
         tau * dot( jump( u ), jump( v ) ) );

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
         << "FAILED: vector interior jump SGBSR action does not match "
         << "matrix-free action.\n";
      return 1;
   }

   std::cout << "SUCCESS: order " << order << " passed.\n";
   return 0;
}

} // namespace

int main()
{
   if ( TestVectorInteriorJumpBsrAction< 2, false >() != 0 )
   {
      return 1;
   }

   if ( TestVectorInteriorJumpBsrAction< 2, true >() != 0 )
   {
      return 1;
   }

   std::cout << "\nVector interior jump BSR action test passed.\n";
   return 0;
}
